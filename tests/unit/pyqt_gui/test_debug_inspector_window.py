from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import QPushButton

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.debug import (
    DebugArtifactRef,
    DebugCursor,
    DebugSnapshot,
    DebugSession,
    LocalDebugSnapshotStore,
)
from openhcs.pyqt_gui.windows.debug_inspector_window import (
    DebugArtifactMaterializeRequest,
    DebugArtifactOpenRequest,
    DebugInspectorWindow,
)
from openhcs.ui.shared.streaming_service import StreamingService

IDENTIFY_PRIMARY_OBJECTS = "IdentifyPrimaryObjects"
DEBUG_GROUP_KEY = "default"
DEBUG_INVOCATION_KEY = "default:0:segment"
DEBUG_SNAPSHOT_ID = "snapshot-1"


def debug_cursor() -> DebugCursor:
    """Return the canonical cursor used by debug-inspector tests."""

    return DebugCursor(
        step_index=1,
        step_scope_id="plate::step-1",
        group_key=DEBUG_GROUP_KEY,
        invocation_key=DEBUG_INVOCATION_KEY,
    )


def debug_snapshot(*, artifact_refs=()) -> DebugSnapshot:
    """Return the canonical snapshot used by debug-inspector tests."""

    return DebugSnapshot(
        snapshot_id=DEBUG_SNAPSHOT_ID,
        cursor=debug_cursor(),
        step_name=IDENTIFY_PRIMARY_OBJECTS,
        callable_name=IDENTIFY_PRIMARY_OBJECTS,
        axis_id="A01",
        output_artifact_refs=artifact_refs,
    )


class QtApplicationHarness:
    """Nominal owner for the QApplication singleton used by GUI smoke tests."""

    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def test_debug_inspector_loads_local_snapshot_store(tmp_path) -> None:
    QtApplicationHarness.app()
    session = DebugSession.create(plate_id="plate")
    snapshot = debug_snapshot()
    store = LocalDebugSnapshotStore.for_session(root_path=tmp_path, session=session)
    store.write_snapshot(snapshot)
    window = DebugInspectorWindow()

    loaded = window.load_snapshot(
        root_path=tmp_path,
        debug_session_id=session.debug_session_id,
        snapshot_id=snapshot.snapshot_id,
    )

    assert loaded == snapshot
    assert IDENTIFY_PRIMARY_OBJECTS in window.title_label.text()


def test_debug_inspector_loads_snapshot_from_store(tmp_path) -> None:
    QtApplicationHarness.app()
    session = DebugSession.create(plate_id="plate")
    snapshot = debug_snapshot()
    store = LocalDebugSnapshotStore.for_session(root_path=tmp_path, session=session)
    store.write_snapshot(snapshot)
    window = DebugInspectorWindow()

    loaded = window.load_snapshot_from_store(
        store=store,
        snapshot_id=snapshot.snapshot_id,
    )

    assert loaded == snapshot
    assert IDENTIFY_PRIMARY_OBJECTS in window.title_label.text()


def test_debug_inspector_emits_typed_artifact_open_request() -> None:
    QtApplicationHarness.app()
    cursor = debug_cursor()
    artifact_ref = DebugArtifactRef(
        kind=ArtifactKind.IMAGE,
        name="Segmented",
        cursor=cursor,
        storage_ref="debug://segmented",
    )
    snapshot = debug_snapshot(artifact_refs=(artifact_ref,))
    window = DebugInspectorWindow()
    requests: list[DebugArtifactOpenRequest] = []
    window.artifact_open_requested.connect(requests.append)
    napari_viewer_type = next(
        viewer_type
        for viewer_type in StreamingService.supported_viewer_types()
        if StreamingService.display_name_for_viewer_type(viewer_type) == "Napari"
    )

    window.set_snapshot(snapshot)
    napari_buttons = [
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "Napari"
    ]
    assert napari_buttons
    napari_buttons[0].click()

    assert requests == [
        DebugArtifactOpenRequest(
            artifact_ref=artifact_ref,
            viewer_type=napari_viewer_type,
        )
    ]


def test_debug_inspector_emits_typed_artifact_export_request() -> None:
    QtApplicationHarness.app()
    artifact_ref = DebugArtifactRef(
        kind=ArtifactKind.MEASUREMENTS,
        name="Measurements",
        cursor=debug_cursor(),
        storage_ref="debug://measurements.csv",
    )
    window = DebugInspectorWindow()
    requests: list[DebugArtifactMaterializeRequest] = []
    window.artifact_export_requested.connect(requests.append)

    window.set_snapshot(debug_snapshot(artifact_refs=(artifact_ref,)))
    export_buttons = [
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "Export"
    ]
    assert export_buttons
    export_buttons[0].click()

    assert requests == [DebugArtifactMaterializeRequest(artifact_ref=artifact_ref)]
