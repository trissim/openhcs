from pathlib import Path

from openhcs.core.config import GlobalPipelineConfig
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.pyqt_gui.widgets.plate_manager import (
    CELLPROFILER_SCOPE_SEPARATOR,
    PlateManagerWidget,
)
from tests.unit.pyqt_gui.test_plate_manager_widget import (
    PlateManagerServiceStub,
    QtApplicationHarness,
    close_widget,
)


def test_plate_manager_expands_multi_cppipe_folder_into_orchestrators(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
    created_orchestrators = []

    def record_orchestrator(
        self,
        plate_path: str,
        *,
        plate_root=None,
        cppipe_path=None,
    ):
        created_orchestrators.append((plate_path, Path(plate_root), Path(cppipe_path)))
        return None

    monkeypatch.setattr(
        PlateManagerWidget,
        "_create_orchestrator_for_plate",
        record_orchestrator,
    )
    widget = PlateManagerWidget(PlateManagerServiceStub())
    plate_root = tmp_path / "plate"
    plate_root.mkdir()
    first_cppipe = plate_root / "first.cppipe"
    second_cppipe = plate_root / "second.cppipe"
    first_cppipe.write_text("Version:5", encoding="utf-8")
    second_cppipe.write_text("Version:5", encoding="utf-8")

    widget.add_plate_callback([plate_root])

    expected_first_scope = f"{plate_root}{CELLPROFILER_SCOPE_SEPARATOR}first"
    expected_second_scope = f"{plate_root}{CELLPROFILER_SCOPE_SEPARATOR}second"
    added = {
        plate["path"]: plate
        for plate in widget.plates
        if plate.get("plate_root") == str(plate_root)
    }
    assert created_orchestrators == [
        (expected_first_scope, plate_root, first_cppipe),
        (expected_second_scope, plate_root, second_cppipe),
    ]
    assert sorted(added) == [expected_first_scope, expected_second_scope]
    assert added[expected_first_scope]["name"] == "plate / first"
    assert added[expected_first_scope]["cppipe_path"] == str(first_cppipe)
    assert added[expected_second_scope]["name"] == "plate / second"
    assert added[expected_second_scope]["cppipe_path"] == str(second_cppipe)
    assert widget.selected_plate_path == expected_first_scope
    close_widget(widget)


def test_plate_manager_prefers_tutorial_start_pipeline_for_multi_cppipe_folder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
    created_orchestrators = []

    def record_orchestrator(
        self,
        plate_path: str,
        *,
        plate_root=None,
        cppipe_path=None,
    ):
        created_orchestrators.append((plate_path, Path(plate_root), Path(cppipe_path)))
        return None

    monkeypatch.setattr(
        PlateManagerWidget,
        "_create_orchestrator_for_plate",
        record_orchestrator,
    )
    widget = PlateManagerWidget(PlateManagerServiceStub())
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    final_cppipe = plate_root / "BBBC022_Analysis_Final.cppipe"
    start_cppipe = plate_root / "BBBC022_Analysis_Start.cppipe"
    final_cppipe.write_text("Version:5", encoding="utf-8")
    start_cppipe.write_text("Version:5", encoding="utf-8")

    widget.add_plate_callback([plate_root])

    expected_start_scope = (
        f"{plate_root}{CELLPROFILER_SCOPE_SEPARATOR}BBBC022_Analysis_Start"
    )
    expected_final_scope = (
        f"{plate_root}{CELLPROFILER_SCOPE_SEPARATOR}BBBC022_Analysis_Final"
    )
    assert created_orchestrators == [
        (expected_start_scope, plate_root, start_cppipe),
        (expected_final_scope, plate_root, final_cppipe),
    ]
    assert widget.selected_plate_path == expected_start_scope
    close_widget(widget)


def test_plate_manager_scoped_cppipe_orchestrator_uses_physical_plate_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "plate"
    plate_root.mkdir()
    scope_id = f"{plate_root}{CELLPROFILER_SCOPE_SEPARATOR}first"

    widget._create_orchestrator_for_plate(scope_id, plate_root=plate_root)

    orchestrator = ObjectStateRegistry.get_object(scope_id)
    assert orchestrator is not None
    assert orchestrator.plate_path == plate_root
    close_widget(widget)
