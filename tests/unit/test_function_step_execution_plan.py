from pathlib import Path

from openhcs.constants.constants import VariableComponents
from openhcs.core.steps.function_artifact_materialization import _build_analysis_filename
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan


def noop(image):
    return image


class ContextStub:
    def __init__(self, compiled_plan):
        self.step_plans = {2: compiled_plan}
        self.filemanager = object()
        self.microscope_handler = object()


def _compiled_plan(**overrides):
    plan = {
        "step_name": "measure",
        "axis_id": "A01",
        "input_dir": "/tmp/input",
        "output_dir": "/tmp/output",
        "variable_components": None,
        "group_by": None,
        "func": noop,
        "artifact_inputs": {},
        "artifact_outputs": {},
        "read_backend": "memory",
        "write_backend": "memory",
        "input_memory_type": "numpy",
        "output_memory_type": "numpy",
        "zarr_config": None,
        "gpu_id": 3,
        "pipeline_position": 9,
        "output_plate_root": "/tmp/plate_processed",
        "sub_dir": "images",
        "analysis_results_dir": "/tmp/output_results",
        "input_conversion_dir": "/tmp/converted",
        "input_conversion_backend": "zarr",
        "materialized_output_dir": "/tmp/materialized",
        "materialized_backend": "disk",
        "materialized_plate_root": "/tmp/plate_materialized",
        "materialized_sub_dir": "images",
        "materialized_analysis_results_dir": "/tmp/materialized_results",
    }
    plan.update(overrides)
    return plan


def test_execution_plan_snapshots_compiled_mapping_without_raw_backing():
    compiled_plan = _compiled_plan()
    context = ContextStub(compiled_plan)

    plan = FunctionStepExecutionPlan.from_context(context, 2)

    assert not hasattr(plan, "raw")
    assert compiled_plan["variable_components"] is None
    assert plan.variable_components == [VariableComponents.SITE]
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
    get_paths_for_axis = lambda _dir, _backend: ["/tmp/output/A01_site1.tif"]
    plan = FunctionStepExecutionPlan(
        **{**plan.__dict__, "get_paths_for_axis": get_paths_for_axis}
    )

    assert (
        _build_analysis_filename("measurements", plan)
        == "A01_site1_measurements_step7.roi.zip"
    )
