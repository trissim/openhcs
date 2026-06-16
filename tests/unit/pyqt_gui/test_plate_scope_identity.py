from pathlib import Path

from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity


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
