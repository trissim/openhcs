from __future__ import annotations

from pathlib import Path

import pytest

from openhcs.core.config import PipelineConfig
from openhcs.core.function_patterns import normalize_function_pattern
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline

_TRAINING_MODEL_XML = """\
<training-data>
  <min-area>12</min-area>
  <max-area>1200</max-area>
  <cost-threshold>8</cost-threshold>
  <num-control-points>7</num-control-points>
  <max-radius>9</max-radius>
  <max-skel-length>100</max-skel-length>
  <min-path-length>10</min-path-length>
  <max-path-length>90</max-path-length>
  <median-worm-area>300</median-worm-area>
  <overlap-weight>5</overlap-weight>
  <leftover-weight>10</leftover-weight>
  <mean-angles><value>0.1</value><value>0.2</value></mean-angles>
  <radii-from-training><value>1</value><value>2</value></radii-from-training>
  <inv-angles-covariance-matrix>
    <values><value>1</value><value>0</value></values>
    <values><value>0</value><value>1</value></values>
  </inv-angles-covariance-matrix>
</training-data>
"""


def _pipeline_text(model_name: str, *, include_straighten: bool) -> str:
    names_and_types = """\
CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:WormObjectsBinary
    Select the rule criteria:and (file does contain "worms")
    Select the image type:Grayscale image
    Name to assign these images:Intensity
    Select the rule criteria:and (file does contain "intensity")
"""
    untangle = f"""\
UntangleWorms:[module_num:2|variable_revision_number:2|enabled:True]
    Select the input binary image:WormObjectsBinary
    Overlap style:Both
    Name the output overlapping worm objects:OverlappingWorms
    Name the output non-overlapping worm objects:NonOverlappingWorms
    Training set file location:Default Input Folder|stale/original/location
    Training set file name:{model_name}
    Use training set weights?:Yes
    Overlap weight:5.0
    Leftover weight:10.0
    Retain outlines of the overlapping objects?:No
    Outline colormap?:Default
    Name the overlapped outline image:OverlappedWormOutlines
    Retain outlines of the non-overlapping worms?:No
    Name the non-overlapped outlines image:NonoverlappedWormOutlines
    Train or untangle worms?:Untangle
    Minimum area percentile:1.0
    Minimum area factor:0.85
    Maximum area percentile:90.0
    Maximum area factor:1.0
    Minimum length percentile:1.0
    Minimum length factor:0.9
    Maximum length percentile:99.0
    Maximum length factor:1.1
    Maximum cost percentile:90.0
    Maximum cost factor:1.9
    Number of control points:21
    Maximum radius percentile:90.0
    Maximum radius factor:1.0
    Maximum complexity:Process all clusters
    Custom complexity:400
"""
    if not include_straighten:
        return names_and_types + untangle
    straighten = f"""\
StraightenWorms:[module_num:3|variable_revision_number:3|enabled:True]
    Select the input untangled worm objects:NonOverlappingWorms
    Name the output straightened worm objects:StraightenedWorms
    Worm width:20
    Training set file location:Default Input Folder|stale/original/location
    Training set file name:{model_name}
    Image count:1
    Measure intensity distribution?:Yes
    Number of transverse segments:5
    Number of longitudinal stripes:1
    Align worms?:Do not align
    Alignment image:Intensity
    Select an input image to straighten:Intensity
    Name the output straightened image:StraightenedIntensity
"""
    return names_and_types + untangle + straighten


def _step_kwargs(steps: list[FunctionStep], name: str) -> dict[str, object]:
    step = next(candidate for candidate in steps if candidate.name == name)
    invocation = next(normalize_function_pattern(step.func).iter_items())
    return dict(invocation.kwargs)


@pytest.mark.parametrize(
    ("pipeline_name", "model_name", "include_straighten"),
    (
        (
            "ExampleUntangleAndStraightenWorms",
            "WormModel.xml",
            True,
        ),
        (
            "ExampleUntangleWorms",
            "MyWormModel_B01_B24.xml",
            False,
        ),
        (
            "ExampleUntangleWormsBrightField",
            "TrainingSetORO.xml",
            False,
        ),
    ),
)
def test_worm_training_models_lower_to_transportable_public_steps(
    tmp_path: Path,
    pipeline_name: str,
    model_name: str,
    include_straighten: bool,
) -> None:
    source_root = tmp_path / "persistent-source"
    source_root.mkdir()
    cppipe_path = source_root / f"{pipeline_name}.cppipe"
    model_path = source_root / model_name
    cppipe_path.write_text(
        _pipeline_text(model_name, include_straighten=include_straighten),
        encoding="utf-8",
    )
    model_path.write_text(_TRAINING_MODEL_XML, encoding="utf-8")

    steps, pipeline_config = import_cellprofiler_pipeline(cppipe_path)

    assert isinstance(pipeline_config, PipelineConfig)
    assert [step.name for step in steps] == (
        ["UntangleWorms", "StraightenWorms"]
        if include_straighten
        else ["UntangleWorms"]
    )
    untangle_kwargs = _step_kwargs(steps, "UntangleWorms")
    assert untangle_kwargs["min_worm_area"] == 12.0
    assert untangle_kwargs["num_control_points"] == 7
    assert untangle_kwargs["mean_angles"] == (0.1, 0.2)
    if include_straighten:
        assert _step_kwargs(steps, "StraightenWorms")["num_control_points"] == 7

    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    assert model_name not in source
    assert str(cppipe_path) not in source
    assert str(source_root) not in source

    cppipe_path.unlink()
    model_path.unlink()
    namespace: dict[str, object] = {}
    exec(compile(source, f"{pipeline_name}_transport.py", "exec"), namespace)
    restored = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)

    assert _step_kwargs(restored, "UntangleWorms") == untangle_kwargs
    if include_straighten:
        assert _step_kwargs(restored, "StraightenWorms")["num_control_points"] == 7


def test_worm_training_model_resolves_from_explicit_default_input_folder(
    tmp_path: Path,
) -> None:
    pipeline_root = tmp_path / "pipeline"
    source_root = tmp_path / "images"
    pipeline_root.mkdir()
    source_root.mkdir()
    cppipe_path = pipeline_root / "ExampleUntangleWorms.cppipe"
    cppipe_path.write_text(
        _pipeline_text("WormModel.xml", include_straighten=False),
        encoding="utf-8",
    )
    (source_root / "WormModel.xml").write_text(
        _TRAINING_MODEL_XML,
        encoding="utf-8",
    )

    steps, _pipeline_config = import_cellprofiler_pipeline(
        cppipe_path,
        source_root=source_root,
    )

    assert _step_kwargs(steps, "UntangleWorms")["num_control_points"] == 7


def test_active_unreferenced_worm_outline_names_survive_import_and_transport(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "ExampleUntangleWorms.cppipe"
    model_name = "WormModel.xml"
    cppipe_path.write_text(
        _pipeline_text(model_name, include_straighten=False)
        .replace(
            "Retain outlines of the overlapping objects?:No",
            "Retain outlines of the overlapping objects?:Yes",
        )
        .replace(
            "Retain outlines of the non-overlapping worms?:No",
            "Retain outlines of the non-overlapping worms?:Yes",
        ),
        encoding="utf-8",
    )
    (tmp_path / model_name).write_text(_TRAINING_MODEL_XML, encoding="utf-8")

    steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)
    kwargs = _step_kwargs(steps, "UntangleWorms")

    assert kwargs["overlapping_outline_name"] == "OverlappedWormOutlines"
    assert kwargs["nonoverlapping_outline_name"] == "NonoverlappedWormOutlines"

    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    namespace: dict[str, object] = {}
    # Execute the generated transport source to prove its round-trip contract.
    exec(  # noqa: S102
        compile(source, "worm_outline_transport.py", "exec"),
        namespace,
    )
    restored = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)

    assert _step_kwargs(restored, "UntangleWorms") == kwargs
