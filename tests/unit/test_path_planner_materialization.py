from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, ArtifactSpec
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    MaterializedOutputPlan,
)
from openhcs.core.pipeline.path_planner import PathPlanner
from openhcs.core.step_dependencies import StepInputDependencyKind


@dataclass(frozen=True)
class PathConfigStub:
    sub_dir: str
    output_dir_suffix: str = "_processed"
    global_output_folder: str | None = None


def _artifact_planner_stub() -> PathPlanner:
    planner = PathPlanner.__new__(PathPlanner)
    planner.plate_path = Path("/data/plate1")
    planner.cfg = PathConfigStub(sub_dir="images")
    planner.ctx = SimpleNamespace(
        axis_id="A01",
        global_config=SimpleNamespace(materialization_results_path="analysis"),
    )
    planner.plans = {
        2: CompiledStepPlan(
            step_index=2,
            step_scope_id="plate::functionstep_2",
            step_name="identify",
            step_type="FunctionStep",
            axis_id="A01",
        )
    }
    planner.declared = {}
    return planner


def test_materialization_collision_updates_results_dir_and_config():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plate_path = Path("/data/plate1")
    planner.plans = {
        3: CompiledStepPlan(
            step_index=3,
            step_name="materialize",
            step_type="FunctionStep",
            axis_id="A01",
            materialized_output=MaterializedOutputPlan(
                output_dir=Path("/data/plate1_processed/images"),
                backend="disk",
                plate_root="/data/plate1_processed",
                sub_dir="images",
                analysis_results_dir="/data/plate1_processed/images_results",
            ),
            materialization_config=PathConfigStub(sub_dir="images"),
        )
    }
    snapshot = SimpleNamespace(
        index=3,
        name="materialize",
        materialization_config=PathConfigStub(sub_dir="images"),
    )

    planner._resolve_and_update_paths(
        snapshot,
        3,
        Path("/data/plate1_processed/images"),
        "main flow",
    )

    assert snapshot.materialization_config.sub_dir == "images"
    materialized_output = planner.plans[3].materialized_output
    assert materialized_output.output_dir == Path("/data/plate1_processed/images_step3")
    assert materialized_output.sub_dir == "images_step3"
    assert materialized_output.analysis_results_dir == (
        "/data/plate1_processed/images_step3_results"
    )
    assert planner.plans[3].materialization_config.sub_dir == "images_step3"


def test_artifact_output_plans_preserve_declared_kind():
    planner = _artifact_planner_stub()

    outputs = planner._process_artifact_outputs(
        {"nuclei": ArtifactSpec("nuclei", ArtifactKind.OBJECT_LABELS)},
        sid=2,
        output_groups={"nuclei": {None}},
        step_name="identify",
    )

    assert outputs["nuclei"].kind is ArtifactKind.OBJECT_LABELS
    assert planner.declared["nuclei"].kind is ArtifactKind.OBJECT_LABELS


def test_execution_groups_use_normalized_group_by_for_variable_conflicts():
    planner = _artifact_planner_stub()

    def fail_component_lookup(_group_by):
        raise AssertionError("normalized GroupBy.NONE must not query components")

    planner.orchestrator = SimpleNamespace(get_component_keys=fail_component_lookup)
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE, VariableComponents.CHANNEL),
        name="source_bound_cellprofiler_step",
    )

    assert planner._get_execution_groups(snapshot) == [None]


def test_artifact_input_plan_rejects_producer_consumer_kind_mismatch():
    planner = _artifact_planner_stub()
    planner.declared["nuclei"] = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        kind=ArtifactKind.OBJECT_LABELS,
        producer_step_index=1,
        producer_step_name="identify",
    )

    with pytest.raises(ValueError, match="expects measurements"):
        planner._process_artifact_inputs(
            {"nuclei": ArtifactSpec("nuclei", ArtifactKind.MEASUREMENTS)},
            {},
            sid=2,
            step_name="measure",
        )


def test_main_input_dependency_uses_scope_identity_for_step_output_edges():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        0: CompiledStepPlan(
            step_index=0,
            step_scope_id="plate::functionstep_0",
            step_name="load",
            step_type="FunctionStep",
            axis_id="A01",
            output_dir=Path("/data/plate1_processed/images"),
        ),
        1: CompiledStepPlan(
            step_index=1,
            step_scope_id="plate::functionstep_1",
            step_name="measure",
            step_type="FunctionStep",
            axis_id="A01",
        ),
    }
    planner.snapshots_by_index = {
        0: SimpleNamespace(scope_id="plate::functionstep_0"),
        1: SimpleNamespace(scope_id="plate::functionstep_1"),
    }

    dependency = planner._main_input_dependency(
        SimpleNamespace(input_source=None),
        1,
    )

    assert dependency.kind is StepInputDependencyKind.STEP_OUTPUT
    assert dependency.source_step_index == 0
    assert dependency.source_step_scope_id == "plate::functionstep_0"

    input_dir, output_dir = planner._step_io_dirs(dependency, 1)
    assert input_dir == Path("/data/plate1_processed/images")
    assert output_dir == Path("/data/plate1_processed/images")


def test_main_input_dependency_preserves_pipeline_start_edges():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        1: CompiledStepPlan(
            step_index=1,
            step_scope_id="plate::functionstep_1",
            step_name="qc",
            step_type="FunctionStep",
            axis_id="A01",
        )
    }
    planner.initial_input = Path("/data/plate1/images")
    planner.snapshots_by_index = {
        1: SimpleNamespace(scope_id="plate::functionstep_1")
    }
    planner._build_output_path = lambda *_args, **_kwargs: Path(
        "/data/plate1_processed/images"
    )

    dependency = planner._main_input_dependency(
        SimpleNamespace(input_source=InputSource.PIPELINE_START),
        1,
    )

    assert dependency.kind is StepInputDependencyKind.PIPELINE_START
    input_dir, output_dir = planner._step_io_dirs(dependency, 1)
    assert input_dir == Path("/data/plate1/images")
    assert output_dir == Path("/data/plate1_processed/images")
