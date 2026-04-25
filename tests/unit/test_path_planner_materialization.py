from dataclasses import dataclass
from pathlib import Path

from openhcs.core.pipeline.path_planner import PathPlanner


@dataclass(frozen=True)
class PathConfigStub:
    sub_dir: str
    output_dir_suffix: str = "_processed"
    global_output_folder: str | None = None


class StepStub:
    def __init__(self, materialization_config: PathConfigStub):
        self.step_materialization_config = materialization_config


def test_materialization_collision_updates_results_dir_and_config():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plate_path = Path("/data/plate1")
    planner.plans = {
        3: {
            "materialized_output_dir": "/data/plate1_processed/images",
            "materialized_sub_dir": "images",
            "materialized_analysis_results_dir": "/data/plate1_processed/images_results",
            "materialization_config": PathConfigStub(sub_dir="images"),
        }
    }
    step = StepStub(PathConfigStub(sub_dir="images"))

    planner._resolve_and_update_paths(
        step,
        3,
        Path("/data/plate1_processed/images"),
        "main flow",
    )

    assert step.step_materialization_config.sub_dir == "images_step3"
    assert planner.plans[3]["materialized_output_dir"] == (
        "/data/plate1_processed/images_step3"
    )
    assert planner.plans[3]["materialized_sub_dir"] == "images_step3"
    assert planner.plans[3]["materialized_analysis_results_dir"] == (
        "/data/plate1_processed/images_step3_results"
    )
    assert planner.plans[3]["materialization_config"].sub_dir == "images_step3"
