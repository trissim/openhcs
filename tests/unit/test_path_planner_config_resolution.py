from pathlib import Path
from types import SimpleNamespace

from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyPathPlanningConfig,
    LazyProcessingConfig,
    LazyStepMaterializationConfig,
    PathPlanningConfig,
    PipelineConfig,
)
from openhcs.core.pipeline.path_planner import PathPlanner
from openhcs.core.steps.abstract import AbstractStep
from objectstate import ObjectState, ObjectStateRegistry


class _DummyStep(AbstractStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = None

    def process(self, context, step_index):
        pass


def _plan_single_step(
    tmp_path: Path,
    *,
    pipeline_materialization_enabled=None,
    step_materialization_enabled=True,
    step_output_dir_suffix=None,
):
    plate_path = tmp_path / "plate"
    global_output_folder = tmp_path / "outputs"
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix="_analysis",
            global_output_folder=global_output_folder,
        ),
        processing_config=LazyProcessingConfig(
            input_source=InputSource.PIPELINE_START
        ),
        step_materialization_config=LazyStepMaterializationConfig(
            sub_dir="snapshots",
            enabled=pipeline_materialization_enabled,
        ),
    )
    step = _DummyStep(
        step_materialization_config=LazyStepMaterializationConfig(
            enabled=step_materialization_enabled,
            output_dir_suffix=step_output_dir_suffix,
        )
    )

    pipeline_state = ObjectState(
        pipeline_config,
        scope_id=str(plate_path),
        parent_state=ObjectStateRegistry.get_by_scope(""),
    )
    ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
    step_state = ObjectState(
        step,
        scope_id=f"{plate_path}::step_0",
        parent_state=pipeline_state,
    )
    ObjectStateRegistry.register(step_state, _skip_snapshot=True)

    effective_config = GlobalPipelineConfig(
        path_planning_config=PathPlanningConfig(
            output_dir_suffix="_analysis",
            global_output_folder=global_output_folder,
        )
    )
    context = SimpleNamespace(
        input_dir=plate_path / "images",
        plate_path=plate_path,
        step_plans={},
        axis_id="A01",
        global_config=effective_config,
    )

    try:
        PathPlanner(
            context,
            effective_config,
            step_state_map={0: step_state},
        ).plan([step])
        return context.step_plans[0], global_output_folder, plate_path
    finally:
        ObjectStateRegistry.unregister(step_state, _skip_snapshot=True)
        ObjectStateRegistry.unregister(pipeline_state, _skip_snapshot=True)


def test_materialization_inherits_all_pipeline_path_fields(tmp_path):
    plan, global_output_folder, plate_path = _plan_single_step(tmp_path)

    expected_plate_root = global_output_folder / f"{plate_path.name}_analysis"
    assert Path(plan["materialized_plate_root"]) == expected_plate_root
    assert Path(plan["materialized_output_dir"]) == expected_plate_root / "snapshots"
    assert plan["materialized_sub_dir"] == "snapshots"
    assert plan["materialization_config"].output_dir_suffix == "_analysis"
    assert plan["materialization_config"].global_output_folder == global_output_folder
    assert plan["input_source"] == "PIPELINE_START"


def test_materialization_explicit_suffix_overrides_pipeline_value(tmp_path):
    plan, global_output_folder, plate_path = _plan_single_step(
        tmp_path,
        step_output_dir_suffix="_step",
    )

    expected_plate_root = global_output_folder / f"{plate_path.name}_step"
    assert Path(plan["materialized_plate_root"]) == expected_plate_root
    assert Path(plan["materialized_output_dir"]) == expected_plate_root / "snapshots"
    assert plan["materialization_config"].output_dir_suffix == "_step"


def test_materialization_inherits_enabled_from_pipeline_defaults(tmp_path):
    plan, global_output_folder, plate_path = _plan_single_step(
        tmp_path,
        pipeline_materialization_enabled=True,
        step_materialization_enabled=None,
    )

    expected_plate_root = global_output_folder / f"{plate_path.name}_analysis"
    assert Path(plan["materialized_output_dir"]) == expected_plate_root / "snapshots"
    assert plan["materialization_config"].enabled is True
