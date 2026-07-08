from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionStep


def test_pipeline_is_explicit_carrier_not_list_subclass() -> None:
    step = FunctionStep(func=lambda image: image, name="Identity")
    pipeline = Pipeline(steps=[step], name="editor")

    assert not isinstance(pipeline, list)
    assert pipeline.steps == [step]
    assert list(pipeline) == [step]


def test_pipeline_mutation_updates_explicit_steps() -> None:
    first = FunctionStep(func=lambda image: image, name="First")
    second = FunctionStep(func=lambda image: image, name="Second")
    pipeline = Pipeline(steps=[first], name="editor")

    pipeline.append(second)

    assert pipeline.steps == [first, second]
