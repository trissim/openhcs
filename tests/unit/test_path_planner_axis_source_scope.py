from pathlib import Path
from types import SimpleNamespace

from polystore.virtual_workspace import SourcePixelRef

from openhcs.core.pipeline.path_planner import PathPlanner
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection


def test_path_planner_scopes_source_metadata_to_compilation_axis() -> None:
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": SourcePixelRef(
                "disk", "/source/A01_w1.tif"
            ),
            "A12_s002_w1_z001_t001.tif": SourcePixelRef(
                "disk", "/source/A12_w1.tif"
            ),
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {
                "well": "A01",
                "site": "1",
                "channel": "1",
            },
            "A12_s002_w1_z001_t001.tif": {
                "well": "A12",
                "site": "2",
                "channel": "1",
            },
        },
    )
    context = SimpleNamespace(
        axis_id="A01",
        input_dir="/workspace",
        plate_path=Path("/workspace"),
    )
    session = SimpleNamespace(
        context=context,
        global_config=SimpleNamespace(
            path_planning_config=SimpleNamespace(),
            vfs_config=SimpleNamespace(),
        ),
        plans={},
        orchestrator=SimpleNamespace(),
        realized_source_metadata=tuple(
            projection.filtered_by_axis(
                axis_id=context.axis_id,
            ).source_metadata_by_path.values()
        ),
        step_count=0,
    )

    planner = PathPlanner(session)

    assert planner.session.realized_source_metadata == (
        {
            "well": "A01",
            "site": "1",
            "channel": "1",
        },
    )
