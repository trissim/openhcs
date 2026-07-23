import openhcs.core.pipeline as pipeline_module
from openhcs.core.steps.function_step import FunctionStep


def test_core_pipeline_module_has_no_execution_wrapper() -> None:
    assert "Pipeline" not in pipeline_module.__all__
    assert not hasattr(pipeline_module, "Pipeline")


def test_pipeline_definition_is_a_plain_mutable_step_list() -> None:
    first = FunctionStep(name="first")
    second = FunctionStep(name="second")
    pipeline_steps = [first]

    pipeline_steps.append(second)
    pipeline_steps[0] = second

    assert pipeline_steps == [second, second]
