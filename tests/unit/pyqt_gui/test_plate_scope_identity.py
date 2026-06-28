from pathlib import Path

from openhcs.pyqt_gui.services.plate_scope_identity import (
    PipelineScopeIdentity,
    PlateScopeIdentity,
)


def test_cellprofiler_pipeline_scope_keeps_cppipe_identity_inside_plate_segment(
    tmp_path: Path,
) -> None:
    plate_root = tmp_path / "AdvancedSegmentation"
    cppipe_path = plate_root / "BBBC022 Analysis Final.cppipe"

    identity = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        cppipe_path,
    )
    parsed = PlateScopeIdentity.from_scope_id(identity.scope_id)

    assert "::" not in identity.scope_id
    assert parsed.plate_root == plate_root
    assert parsed.cppipe_path == cppipe_path
    assert parsed.display_name == "AdvancedSegmentation / BBBC022 Analysis Final"


def test_pipeline_scope_identity_parses_cppipe_plate_scope(tmp_path: Path) -> None:
    plate_root = tmp_path / "AdvancedSegmentation"
    cppipe_path = plate_root / "BBBC022 Analysis Final.cppipe"
    plate_identity = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        cppipe_path,
    )

    pipeline_identity = PipelineScopeIdentity.from_plate_scope(
        plate_identity.scope_id,
    )
    parsed = PipelineScopeIdentity.from_scope_id(pipeline_identity.scope_id)

    assert PipelineScopeIdentity.matches(pipeline_identity.scope_id)
    assert parsed.plate_scope == plate_identity.scope_id
