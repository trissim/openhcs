"""Public FunctionStep round-trip coverage for CellProfiler ImageMath."""

from pathlib import Path

from openhcs.constants import GroupBy, VariableComponents
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline


def _image_math_public_kwargs(pipeline_steps) -> tuple[dict[str, object], ...]:
    image_math_steps = tuple(
        step for step in pipeline_steps if step.name == "ImageMath"
    )
    assert len(image_math_steps) == 2
    assert tuple(step.processing_config.group_by for step in image_math_steps) == (
        GroupBy.CHANNEL,
        GroupBy.NONE,
    )
    return tuple(
        invocation.kwargs_dict
        for invocation in normalize_function_pattern(
            image_math_steps[1].func
        ).iter_items()
    )


def test_3d_monolayer_image_math_round_trips_public_function_steps() -> None:
    cppipe_path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "native_refs"
        / "official30_scoped_rows"
        / "CellProfiler_tutorials_cp_tutorial_3d_monolayer_samples_first2wells"
        / "native_cellprofiler_headless"
        / "3d_monolayer_final.cppipe"
    )
    expected_kwargs = (
        {
            "factors": (1.0, 1.0, 1.0),
            "select_the_second_image": "origMemb",
            "select_the_third_image": "origMito",
            "name_the_output_image": "Monolayer",
        },
    )

    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)

    assert _image_math_public_kwargs(pipeline_steps) == expected_kwargs

    source = FunctionStepTransportAuthority.source_from_pipeline(pipeline_steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "3d_monolayer_final.py", "exec"), namespace)
    reconstructed_steps = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )

    assert _image_math_public_kwargs(reconstructed_steps) == expected_kwargs


def test_single_channel_image_math_retains_inherited_channel_grouping() -> None:
    cppipe_path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "native_refs"
        / "official30_scoped_rows"
        / "ExampleIlluminationCorrection_ExampleIlluminationCorrection_Example3_wells_include_first1"
        / "native_cellprofiler_headless"
        / "ExampleIlluminationCorrection_Example3.cppipe"
    )
    pipeline_steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)
    image_math_steps = tuple(
        step for step in pipeline_steps if step.name == "ImageMath"
    )

    assert len(image_math_steps) == 1
    assert image_math_steps[0].processing_config.variable_components == [
        VariableComponents.SITE
    ]
    assert image_math_steps[0].processing_config.group_by is GroupBy.CHANNEL
