from pathlib import Path

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
from openhcs.core.steps.function_artifact_materialization import _build_analysis_filename
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.core.steps.function_runtime import _select_artifact_plan_for_component


def noop(image):
    return image


class ContextStub:
    def __init__(self, compiled_plan):
        self.step_plans = {2: compiled_plan}
        self.filemanager = object()
        self.microscope_handler = object()


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
    compiled_plan = _compiled_plan()
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


def test_build_analysis_filename_uses_pipeline_position_for_image_derived_name():
    plan = FunctionStepExecutionPlan.from_context(
        ContextStub(_compiled_plan(pipeline_position=7)),
        2,
    )
    def get_paths_for_axis(_dir, _backend):
        return ["/tmp/output/A01_site1.tif"]

    plan = FunctionStepExecutionPlan(
        **{**plan.__dict__, "get_paths_for_axis": get_paths_for_axis}
    )

    assert (
        _build_analysis_filename("measurements", plan)
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

    selected = _select_artifact_plan_for_component(
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
