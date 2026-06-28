from pathlib import Path

from PyQt6.QtCore import Qt

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.collection_containers import RootState
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.plate_scope_identity import PlateScopeIdentity
from openhcs.pyqt_gui.services.plate_manager_row import PlateManagerRow
from openhcs.pyqt_gui.widgets.plate_manager import (
    PlateManagerWidget,
    ROOT_SCOPE_ID,
)
from openhcs.pyqt_gui.widgets.shared.services.plate_manager_workflows import (
    PlateManagerCodeWorkflow,
    PlateManagerDeletionWorkflow,
)
from tests.unit.pyqt_gui.test_plate_manager_widget import (
    PlateManagerServiceStub,
    QtApplicationHarness,
    close_widget,
)


def test_plate_manager_registers_multi_cppipe_folder_as_logical_pipeline_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
    created_orchestrators = []
    create_orchestrator = PlateManagerWidget._create_orchestrator_for_plate

    def record_orchestrator(
        self,
        plate_path: str,
        *,
        plate_root=None,
        cppipe_path=None,
    ):
        created_orchestrators.append((plate_path, Path(plate_root), Path(cppipe_path)))
        return create_orchestrator(
            self,
            plate_path,
            plate_root=plate_root,
            cppipe_path=cppipe_path,
        )

    monkeypatch.setattr(
        PlateManagerWidget,
        "_create_orchestrator_for_plate",
        record_orchestrator,
    )
    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "plate"
    plate_root.mkdir()
    first_cppipe = plate_root / "first.cppipe"
    second_cppipe = plate_root / "second.cppipe"
    first_cppipe.write_text("Version:5", encoding="utf-8")
    second_cppipe.write_text("Version:5", encoding="utf-8")

    widget.add_plate_callback([plate_root])

    first_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        first_cppipe,
    ).scope_id
    second_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        second_cppipe,
    ).scope_id
    assert "::" not in first_scope
    assert "::" not in second_scope
    added = {
        plate.scope_id: plate
        for plate in widget.plates
        if plate.plate_root == str(plate_root)
    }
    assert created_orchestrators == [
        (first_scope, plate_root, first_cppipe),
        (second_scope, plate_root, second_cppipe),
    ]
    assert sorted(added) == [first_scope, second_scope]
    assert added[first_scope].name == "plate / first"
    assert added[first_scope].cppipe_path == str(first_cppipe)
    assert added[second_scope].name == "plate / second"
    assert added[second_scope].cppipe_path == str(second_cppipe)
    assert widget.selected_plate_path == first_scope
    close_widget(widget)


def test_plate_manager_prefers_tutorial_final_pipeline_for_multi_cppipe_folder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
    created_orchestrators = []
    create_orchestrator = PlateManagerWidget._create_orchestrator_for_plate

    def record_orchestrator(
        self,
        plate_path: str,
        *,
        plate_root=None,
        cppipe_path=None,
    ):
        created_orchestrators.append((plate_path, Path(plate_root), Path(cppipe_path)))
        return create_orchestrator(
            self,
            plate_path,
            plate_root=plate_root,
            cppipe_path=cppipe_path,
        )

    monkeypatch.setattr(
        PlateManagerWidget,
        "_create_orchestrator_for_plate",
        record_orchestrator,
    )
    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    final_cppipe = plate_root / "BBBC022_Analysis_Final.cppipe"
    start_cppipe = plate_root / "BBBC022_Analysis_Start.cppipe"
    final_cppipe.write_text("Version:5", encoding="utf-8")
    start_cppipe.write_text("Version:5", encoding="utf-8")

    widget.add_plate_callback([plate_root])

    start_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        start_cppipe,
    ).scope_id
    final_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        final_cppipe,
    ).scope_id
    assert created_orchestrators == [
        (start_scope, plate_root, start_cppipe),
        (final_scope, plate_root, final_cppipe),
    ]
    assert widget.selected_plate_path == final_scope
    close_widget(widget)


def test_plate_manager_cppipe_add_keeps_visible_selection_on_logical_scope(
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    final_cppipe = plate_root / "BBBC022_Analysis_Final.cppipe"
    start_cppipe = plate_root / "BBBC022_Analysis_Start.cppipe"
    final_cppipe.write_text("Version:5", encoding="utf-8")
    start_cppipe.write_text("Version:5", encoding="utf-8")

    try:
        widget.add_plate_callback([plate_root])

        final_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            final_cppipe,
        ).scope_id
        current_item = widget.item_list.currentItem()

        assert widget.selected_plate_path == final_scope
        assert current_item is not None
        assert current_item.data(Qt.ItemDataRole.UserRole).scope_id == final_scope
    finally:
        close_widget(widget)
        ObjectStateRegistry.clear()


def test_plate_manager_refresh_preserves_selection_without_reemitting_plate_selected(
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    final_cppipe = plate_root / "BBBC022_Analysis_Final.cppipe"
    final_cppipe.write_text("Version:5", encoding="utf-8")

    try:
        widget.add_plate_callback([plate_root])
        emissions = []
        widget.plate_selected.connect(emissions.append)

        widget.update_item_list()

        assert emissions == []
    finally:
        close_widget(widget)
        ObjectStateRegistry.clear()


def test_plate_manager_delete_selected_cppipe_selects_remaining_scope(
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "BeginnerSegmentation"
    plate_root.mkdir()
    start_cppipe = plate_root / "segmentation_start.cppipe"
    final_cppipe = plate_root / "segmentation_final.cppipe"
    start_cppipe.write_text("Version:5", encoding="utf-8")
    final_cppipe.write_text("Version:5", encoding="utf-8")

    try:
        widget.add_plate_callback([plate_root])
        start_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            start_cppipe,
        ).scope_id
        final_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            final_cppipe,
        ).scope_id
        emissions = []
        widget.plate_selected.connect(emissions.append)

        widget.selected_plate_path = final_scope
        PlateManagerDeletionWorkflow(widget).delete(
            [PlateManagerRow.from_scope(final_scope)]
        )

        remaining_scope_ids = [row.scope_id for row in widget.plates]
        assert remaining_scope_ids == [start_scope]
        assert widget.selected_plate_path == start_scope
        assert emissions == [start_scope]
    finally:
        close_widget(widget)
        ObjectStateRegistry.clear()


def test_plate_manager_delete_action_moves_pipeline_editor_to_remaining_cppipe(
    tmp_path: Path,
) -> None:
    from openhcs.pyqt_gui.widgets.pipeline_editor import PipelineEditorWidget

    QtApplicationHarness.app()
    ObjectStateRegistry.clear()

    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    editor = PipelineEditorWidget(service_adapter)
    widget.pipeline_editor = editor
    widget.plate_selected.connect(editor.set_current_plate)
    plate_root = tmp_path / "BeginnerSegmentation"
    plate_root.mkdir()
    start_cppipe = plate_root / "segmentation_start.cppipe"
    final_cppipe = plate_root / "segmentation_final.cppipe"
    start_cppipe.write_text("Version:5", encoding="utf-8")
    final_cppipe.write_text("Version:5", encoding="utf-8")

    try:
        widget.add_plate_callback([plate_root])
        start_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            start_cppipe,
        ).scope_id
        final_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            final_cppipe,
        ).scope_id

        assert widget.selected_plate_path == final_scope
        assert editor.current_plate == final_scope

        widget.action_delete()

        assert [row.scope_id for row in widget.plates] == [start_scope]
        assert widget.selected_plate_path == start_scope
        assert editor.current_plate == start_scope
    finally:
        editor.close()
        close_widget(widget)
        ObjectStateRegistry.clear()


def test_plate_manager_normalizes_persisted_multi_cppipe_scope_to_logical_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
    ObjectStateRegistry.clear()

    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    root_state = ObjectState(object_instance=RootState(), scope_id=ROOT_SCOPE_ID)
    ObjectStateRegistry.register(root_state, _skip_snapshot=True)
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    final_cppipe = plate_root / "BBBC022_Analysis_Final.cppipe"
    start_cppipe = plate_root / "BBBC022_Analysis_Start.cppipe"
    final_cppipe.write_text("Version:5", encoding="utf-8")
    start_cppipe.write_text("Version:5", encoding="utf-8")
    root_state.update_parameter("orchestrator_scope_ids", [str(plate_root)])

    try:
        widget = PlateManagerWidget(service_adapter)

        start_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            start_cppipe,
        ).scope_id
        final_scope = PlateScopeIdentity.from_cellprofiler_pipeline(
            plate_root,
            final_cppipe,
        ).scope_id
        normalized_root_state = ObjectStateRegistry.get_by_scope(ROOT_SCOPE_ID)
        assert normalized_root_state.parameters["orchestrator_scope_ids"] == [
            start_scope,
            final_scope,
        ]
        assert (
            ObjectStateRegistry.get_object(start_scope)
            .input_workspace_preparation.selected_pipeline_path
            == start_cppipe
        )
        assert (
            ObjectStateRegistry.get_object(final_scope)
            .input_workspace_preparation.selected_pipeline_path
            == final_cppipe
        )
        close_widget(widget)
    finally:
        ObjectStateRegistry.clear()


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
    cppipe_path = plate_root / "first.cppipe"
    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        cppipe_path,
    ).scope_id

    widget._create_orchestrator_for_plate(scope_id, plate_root=plate_root)

    orchestrator = ObjectStateRegistry.get_object(scope_id)
    assert orchestrator is not None
    assert orchestrator.plate_path == plate_root
    close_widget(widget)


def test_plate_manager_code_mode_cppipe_scope_preserves_import_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    ObjectStateRegistry.clear()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_item_list", lambda self: None)
    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "BeginnerSegmentation"
    plate_root.mkdir()
    cppipe_path = plate_root / "segmentation_final.cppipe"
    cppipe_path.write_text("Version:5", encoding="utf-8")
    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        cppipe_path,
    ).scope_id

    try:
        PlateManagerCodeWorkflow(widget).sync_plate_entries((scope_id,))

        orchestrator = ObjectStateRegistry.get_object(scope_id)
        assert orchestrator is not None
        assert orchestrator.plate_path == plate_root
        assert orchestrator.input_workspace_preparation is not None
        assert orchestrator.input_workspace_preparation.selected_path == plate_root
        assert (
            orchestrator.input_workspace_preparation.selected_pipeline_path
            == cppipe_path
        )
    finally:
        close_widget(widget)
        ObjectStateRegistry.clear()


def test_plate_manager_opens_cppipe_config_with_logical_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    ObjectStateRegistry.clear()

    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    cppipe_path = plate_root / "BBBC022_Analysis_Final.cppipe"
    cppipe_path.write_text("Version:5", encoding="utf-8")
    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        cppipe_path,
    ).scope_id
    widget._create_orchestrator_for_plate(
        scope_id,
        plate_root=plate_root,
        cppipe_path=cppipe_path,
    )
    monkeypatch.setattr(
        widget,
        "get_selected_items",
        lambda: [
            PlateManagerRow.from_scope(
                scope_id,
                cppipe_path=str(cppipe_path),
            )
        ],
    )

    captured = {}

    class FakeConfigWindow:
        def __init__(
            self,
            config_class,
            current_config,
            on_save_callback,
            color_scheme=None,
            parent=None,
            scope_id=None,
        ):
            captured["config_class"] = config_class
            captured["current_config"] = current_config
            captured["on_save_callback"] = on_save_callback
            captured["scope_id"] = scope_id

        def show(self):
            captured["shown"] = True

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    monkeypatch.setattr(
        "openhcs.pyqt_gui.widgets.plate_manager.ConfigWindow",
        FakeConfigWindow,
    )
    emissions = []
    widget.orchestrator_config_changed.connect(
        lambda emitted_scope, config: emissions.append((emitted_scope, config))
    )

    try:
        widget.action_edit_config()

        assert captured["config_class"] is PipelineConfig
        assert captured["scope_id"] == scope_id

        new_config = PipelineConfig()
        captured["on_save_callback"](new_config)

        assert widget.plate_configs[scope_id] is new_config
        assert str(plate_root) not in widget.plate_configs
        assert emissions[-1][0] == scope_id
    finally:
        close_widget(widget)
        ObjectStateRegistry.clear()


def test_plate_manager_cppipe_row_preview_uses_logical_config_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    QtApplicationHarness.app()
    monkeypatch.setattr(PlateManagerWidget, "setup_ui", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "setup_connections", lambda self: None)
    monkeypatch.setattr(PlateManagerWidget, "update_button_states", lambda self: None)
    ObjectStateRegistry.clear()

    service_adapter = PlateManagerServiceStub()
    ensure_global_config_context(GlobalPipelineConfig, service_adapter.global_config)
    widget = PlateManagerWidget(service_adapter)
    plate_root = tmp_path / "AdvancedSegmentation"
    plate_root.mkdir()
    cppipe_path = plate_root / "BBBC022_Analysis_Final.cppipe"
    cppipe_path.write_text("Version:5", encoding="utf-8")
    scope_id = PlateScopeIdentity.from_cellprofiler_pipeline(
        plate_root,
        cppipe_path,
    ).scope_id
    widget._create_orchestrator_for_plate(
        scope_id,
        plate_root=plate_root,
        cppipe_path=cppipe_path,
    )

    try:
        rendered = widget._format_plate_item_with_preview_text(
            PlateManagerRow.from_scope(
                scope_id,
                cppipe_path=str(cppipe_path),
            )
        )

        preview_paths = {
            segment.field_path for segment in rendered.layout.preview_segments
        }
        assert "num_workers" in preview_paths
        assert "vfs_config.materialization_backend" in preview_paths
        assert rendered.layout.detail_line == scope_id
    finally:
        close_widget(widget)
        ObjectStateRegistry.clear()
