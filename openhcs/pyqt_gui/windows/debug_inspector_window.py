"""PyQt debug inspector window for OpenHCS debug snapshots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from openhcs.core.debug import (
    DebugArtifactRef,
    DebugSnapshot,
    DebugSnapshotStore,
    LocalDebugSnapshotStore,
)
from openhcs.core.config import StreamingConfig
from openhcs.core.debug_views import (
    DebugViewModel,
    DebugViewSection,
    DebugViewSectionDeclarationBase,
    DebugViewTable,
    DebugViewTableProjectionDeclarationBase,
)
from openhcs.interop.cellprofiler.debug_views import CellProfilerDebugView
from pyqt_reactive.widgets.shared import ActionTabSpec, ActionTabbedWindowBody


@dataclass(frozen=True, slots=True)
class DebugArtifactOpenRequest:
    """Typed request to open one debug artifact in a registered viewer."""

    artifact_ref: DebugArtifactRef
    viewer_type: str


@dataclass(frozen=True, slots=True)
class DebugArtifactMaterializeRequest:
    """Typed request to export one debug artifact through the host GUI."""

    artifact_ref: DebugArtifactRef


@dataclass(frozen=True, slots=True)
class DebugArtifactActionsModel:
    """Viewer-action projection for one debug snapshot."""

    artifact_refs: tuple[DebugArtifactRef, ...]
    viewer_types: tuple[str, ...]

    @classmethod
    def from_snapshot(cls, snapshot: DebugSnapshot) -> "DebugArtifactActionsModel":
        return cls(
            artifact_refs=(
                snapshot.output_artifact_refs
                + snapshot.preview_refs
                + snapshot.input_artifact_refs
            ),
            viewer_types=StreamingConfig.supported_config_keys(),
        )

    @property
    def has_actions(self) -> bool:
        return bool(self.artifact_refs and self.viewer_types)


class DebugInspectorWindow(QDialog):
    """Window that renders a renderer-independent debug view model."""

    artifact_open_requested = pyqtSignal(object)
    artifact_export_requested = pyqtSignal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        snapshot_renderer: Callable[[DebugSnapshot], DebugViewModel] | None = None,
    ) -> None:
        super().__init__(parent)
        self.snapshot_renderer = (
            snapshot_renderer or self._cellprofiler_snapshot_renderer
        )
        self.current_snapshot: DebugSnapshot | None = None
        self.setWindowTitle("Debug Inspector")
        self.setModal(False)
        self.resize(900, 650)

        self._layout = QVBoxLayout(self)
        self.title_label = QLabel("No debug snapshot selected")
        self.title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._layout.addWidget(self.title_label)

        self.tab_body = ActionTabbedWindowBody(parent=self)
        self._layout.addWidget(self.tab_body, 1)

    def set_snapshot(self, snapshot: DebugSnapshot) -> None:
        self.current_snapshot = snapshot
        self.set_view_model(self.snapshot_renderer(snapshot))

    def set_inspection_view_model(self, view_model: DebugViewModel) -> None:
        """Render a non-snapshot debug inspection view."""

        self.current_snapshot = None
        self.set_view_model(view_model)

    def load_snapshot(
        self,
        *,
        root_path: str | Path,
        debug_session_id: str,
        snapshot_id: str,
    ) -> DebugSnapshot:
        """Load and render one snapshot from the local debug snapshot store."""

        snapshot = LocalDebugSnapshotStore(
            root_path=Path(root_path),
            debug_session_id=debug_session_id,
        ).read_snapshot(snapshot_id)
        self.set_snapshot(snapshot)
        return snapshot

    def load_snapshot_from_store(
        self,
        *,
        store: DebugSnapshotStore,
        snapshot_id: str,
    ) -> DebugSnapshot:
        """Load and render one snapshot from any debug snapshot store."""

        snapshot = store.read_snapshot(snapshot_id)
        self.set_snapshot(snapshot)
        return snapshot

    def set_view_model(self, view_model: DebugViewModel) -> None:
        self.title_label.setText(view_model.title)
        self._replace_tab_body()
        for section in view_model.sections:
            self.tab_body.add_tab(
                ActionTabSpec(
                    label=section.title,
                    content=self._section_widget(section),
                    actions=self._section_actions_widget(section),
                )
            )

    @staticmethod
    def _cellprofiler_snapshot_renderer(snapshot: DebugSnapshot) -> DebugViewModel:
        renderer = CellProfilerDebugView.for_module(snapshot.callable_name)
        return renderer.build_view_model(snapshot)

    def _replace_tab_body(self) -> None:
        self._layout.removeWidget(self.tab_body)
        self.tab_body.deleteLater()
        self.tab_body = ActionTabbedWindowBody(parent=self)
        self._layout.addWidget(self.tab_body, 1)

    def _section_widget(self, section: DebugViewSection) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        section_declaration = DebugViewSectionDeclarationBase.for_kind(section.kind)
        if section.table is not None:
            if not section.table.rows:
                layout.addWidget(
                    QLabel(section.table.empty_message or section_declaration.empty_message())
                )
            layout.addWidget(self._table_widget(section.table))
        if section.text is not None:
            text = QTextEdit()
            text.setReadOnly(True)
            text.setPlainText(section.text)
            layout.addWidget(text)
        if section.table is None and section.text is None:
            layout.addWidget(QLabel(section_declaration.empty_message()))
        return widget

    def _section_actions_widget(self, section: DebugViewSection) -> QWidget | None:
        if self.current_snapshot is None or section.table is None:
            return None
        if section.table.projection is None:
            return None
        table_declaration = DebugViewTableProjectionDeclarationBase.for_projection(
            section.table.projection
        )
        if not table_declaration.supports_artifact_actions:
            return None
        return self._artifact_actions_widget(self.current_snapshot)

    @staticmethod
    def _table_widget(table: DebugViewTable) -> QTableWidget:
        widget = QTableWidget(len(table.rows), len(table.columns))
        widget.setHorizontalHeaderLabels(table.columns)
        for row_index, row in enumerate(table.rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setFlags(
                    item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                widget.setItem(row_index, column_index, item)
        widget.resizeColumnsToContents()
        widget.resizeRowsToContents()
        return widget

    def _artifact_actions_widget(self, snapshot: DebugSnapshot) -> QGroupBox | None:
        actions_model = DebugArtifactActionsModel.from_snapshot(snapshot)
        if not actions_model.has_actions:
            return None
        group = QGroupBox("Open Artifacts")
        layout = QVBoxLayout(group)
        for artifact_ref in actions_model.artifact_refs:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{artifact_ref.kind.value}: {artifact_ref.name}"))
            export_button = QPushButton("Export")
            export_button.clicked.connect(
                lambda _=False, ref=artifact_ref: self.request_export_artifact(ref)
            )
            row.addWidget(export_button)
            for viewer_type in actions_model.viewer_types:
                button = QPushButton(
                    StreamingConfig.display_name_for_config_key(viewer_type)
                )
                button.clicked.connect(
                    lambda _=False, ref=artifact_ref, target=viewer_type: (
                        self.request_open_artifact(ref, target)
                    )
                )
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
        return group

    def request_open_artifact(
        self,
        artifact_ref: DebugArtifactRef,
        viewer_type: str,
    ) -> None:
        """Emit a typed request for the host GUI to stream a debug artifact."""

        self.artifact_open_requested.emit(
            DebugArtifactOpenRequest(
                artifact_ref=artifact_ref,
                viewer_type=viewer_type,
            )
        )

    def request_export_artifact(self, artifact_ref: DebugArtifactRef) -> None:
        """Emit a typed request for the host GUI to materialize a debug artifact."""

        self.artifact_export_requested.emit(
            DebugArtifactMaterializeRequest(artifact_ref=artifact_ref)
        )


def is_debug_inspector_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_debug_inspector_export(name, value)
)
