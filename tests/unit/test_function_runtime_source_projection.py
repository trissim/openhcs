from pathlib import Path

from openhcs.core.steps.function_runtime import VirtualWorkspaceSourceProjection


def test_virtual_workspace_pipeline_start_files_preserve_virtual_identity(tmp_path: Path) -> None:
    """Pipeline-start source resolution must not collapse per-well virtual files."""

    plate_path = tmp_path / "plate"
    real_path = tmp_path / "source" / "image.png"
    virtual_a = plate_path / "W001_s001_w1_z001_t001.png"
    virtual_b = plate_path / "W002_s001_w1_z001_t001.png"
    projection = VirtualWorkspaceSourceProjection(
        source_paths_by_virtual_path={
            virtual_a.name: str(real_path),
            str(virtual_a): str(real_path),
            virtual_b.name: str(real_path),
            str(virtual_b): str(real_path),
        },
        source_metadata_by_path={
            virtual_a.name: {"well": "W001"},
            virtual_b.name: {"well": "W002"},
        },
    )

    assert projection.pipeline_start_files() == (str(virtual_a), str(virtual_b))
    assert projection.source_metadata_for(
        virtual_path=virtual_a.name,
        full_virtual_path=str(virtual_a),
    ) == {"well": "W001"}
    assert projection.source_metadata_for(
        virtual_path=virtual_b.name,
        full_virtual_path=str(virtual_b),
    ) == {"well": "W002"}
