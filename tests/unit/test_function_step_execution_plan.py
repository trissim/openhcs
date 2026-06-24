from pathlib import Path
from types import SimpleNamespace

from openhcs.constants.constants import VariableComponents
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    InputConversionPlan,
    MaterializedOutputPlan,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.step_dependencies import (
    StepInputDependency,
    StepInputDependencyKind,
)
from openhcs.core.steps.function_artifact_materialization import (
    AnalysisOutputDescriptorAuthority,
)
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity
from openhcs.core.steps.function_output_manifest import (
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
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
        variable_components=None,
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
        gpu_id=3,
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


def test_execution_plan_snapshots_compiled_plan_without_raw_backing():
    compiled_plan = _compiled_plan(source_identity_stack_axes=frozenset({"z_index"}))
    context = ContextStub(compiled_plan)

    plan = FunctionStepExecutionPlan.from_context(context, 2)

    assert not hasattr(plan, "raw")
    assert plan.step_scope_id == "plate::functionstep_2"
    assert compiled_plan.variable_components is None
    assert plan.variable_components == [VariableComponents.SITE]
    assert plan.main_input_dependency.kind is StepInputDependencyKind.STEP_OUTPUT
    assert plan.main_input_dependency.source_step_scope_id == "plate::functionstep_1"
    assert plan.source_binding_plan.is_empty
    assert plan.device_id is None
    assert plan.has_input_conversion
    assert plan.input_conversion_dir == Path("/tmp/converted")
    assert plan.input_conversion_original_subdir == "input"
    assert plan.has_materialized_output
    assert plan.materialized_output_dir == Path("/tmp/materialized")
    assert plan.artifact_analysis_output_dir == Path("/tmp/materialized_results")
    assert plan.source_identity_stack_axes == frozenset({"z_index"})


def test_function_step_execution_does_not_prepare_callables_in_hot_path(monkeypatch):
    from openhcs.core.steps import function_execution

    events = []

    class ManifestStub:
        def begin_step(self, plan):
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


def test_build_analysis_filename_uses_pipeline_position_for_image_derived_name():
    context = ContextStub(_compiled_plan(pipeline_position=7))
    plan = FunctionStepExecutionPlan.from_context(context, 2)
    step_output_manifest(context).record_outputs(
        plan,
        (
            ProducedOutputSemantics.from_output(
                plan,
                "/tmp/output/A01_site1.tif",
                FunctionOutputIdentity(
                    component_values={"well": "A01", "site": "1"},
                    extension=".tif",
                    source="test",
                ),
            ),
        ),
    )

    assert (
        AnalysisOutputDescriptorAuthority.build(
            "measurements",
            plan,
            context=context,
        ).filename
        == "A01_site1_measurements_step7.roi.zip"
    )


def test_component_artifact_plan_selection_merges_global_and_group_outputs():
    global_output = ArtifactOutputPlan(
        name="objects",
        path="/tmp/objects",
        kind=ArtifactKind.OBJECT_LABELS,
    )
    grouped_output = ArtifactOutputPlan(
        name="measurements",
        path="/tmp/measurements/A01",
        kind=ArtifactKind.MEASUREMENTS,
    )

    selected = ComponentArtifactPlans._select_plan_for_component(
        {
            None: {"objects": global_output},
            "A01": {"measurements": grouped_output},
        },
        "A01",
        {},
    )

    assert selected == {
        "objects": global_output,
        "measurements": grouped_output,
    }
