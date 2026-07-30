from __future__ import annotations

import json
from collections.abc import Iterable
from enum import IntEnum
from pathlib import Path

import napari
import numpy as np
from napari.layers import Shapes
from napari.layers.base import ActionType
from napari.layers.shapes._shapes_constants import ShapeType
from napari.utils._proxies import PublicOnlyProxy
from qtpy import QtCore
from qtpy import QtWidgets as QtW
from roifile import roiread, roiwrite

from openhcs.napari_roi_manager._dataclasses import RoiData, RoiTuple
from openhcs.napari_roi_manager.ij._convert import roi_to_shape, shape_to_roi
from openhcs.napari_roi_manager.widgets._dialogs import QCustomDialog


def _native_object(value):
    """Return the native Napari owner carried by a plugin public proxy."""
    if isinstance(value, PublicOnlyProxy):
        return value.__wrapped__
    return value


class QRoiManagerButtons(QtW.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtW.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self._add_roi_btn = QtW.QPushButton("Add", self)
        self._add_roi_btn.setToolTip("Register the selected native shape as an ROI.")
        self._remove_roi_btn = QtW.QPushButton("Remove", self)
        self._remove_roi_btn.setToolTip("Remove selected ROIs from the native layer.")
        self._specify_btn = QtW.QPushButton("Specify", self)
        self._specify_btn.setToolTip("Specify a rectangle ROI.")
        self._load_roiset_btn = QtW.QPushButton("Load", self)
        self._load_roiset_btn.setToolTip("Load ROIs into the native layer.")
        self._save_roiset_btn = QtW.QPushButton("Save", self)
        self._save_roiset_btn.setToolTip("Save ROIs from the native layer.")
        self._new_set_btn = QtW.QPushButton("New Set", self)
        self._new_set_btn.setToolTip("Create and bind a new native Shapes layer.")

        self._text_group = QtW.QGroupBox("ROI Text")
        text_layout = QtW.QVBoxLayout(self._text_group)
        text_layout.setContentsMargins(2, 8, 2, 4)
        self._text_feature_name = QtW.QComboBox()
        self._text_feature_name.setToolTip("Native feature used for ROI text.")
        self._text_font_size = QtW.QSpinBox()
        self._text_font_size.setRange(4, 64)
        self._text_font_size.setValue(9)
        self._text_font_size.setToolTip("Font size for the ROI text.")
        self._text_font_size.setSuffix(" pt")
        text_layout.addWidget(self._text_feature_name)
        text_layout.addWidget(self._text_font_size)

        self._show_all_checkbox = QtW.QCheckBox("Show All", self)
        self._show_all_checkbox.setChecked(True)

        layout.addWidget(self._add_roi_btn)
        layout.addWidget(self._remove_roi_btn)
        layout.addWidget(self._specify_btn)
        layout.addWidget(self._load_roiset_btn)
        layout.addWidget(self._save_roiset_btn)
        layout.addWidget(self._new_set_btn)
        layout.addWidget(self._text_group)
        layout.addWidget(self._show_all_checkbox)

        self.setFixedWidth(115)

    def set_layer_available(self, available: bool) -> None:
        for widget in (
            self._add_roi_btn,
            self._remove_roi_btn,
            self._specify_btn,
            self._save_roiset_btn,
            self._text_group,
            self._show_all_checkbox,
        ):
            widget.setEnabled(available)


class RoiTableColumn(IntEnum):
    """Columns projected directly from one native Napari Shapes layer."""

    def __new__(cls, value: int, header: str):
        member = int.__new__(cls, value)
        member._value_ = value
        member.header = header
        return member

    NAME = (0, "name")
    SHAPE_TYPE = (1, "type")


class QRoiTableModel(QtCore.QAbstractTableModel):
    """Virtual table projection over the authoritative native Shapes layer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer: Shapes | None = None

    def bind_layer(self, layer: Shapes | None) -> None:
        """Bind a native layer without copying its geometry or feature state."""

        if layer is self._layer:
            self.refresh()
            return
        self.beginResetModel()
        self._layer = layer
        self.endResetModel()

    def refresh(self) -> None:
        """Invalidate the projection after the native owner emits a change."""

        self.beginResetModel()
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        if parent.isValid() or self._layer is None:
            return 0
        return len(self._layer.data)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(RoiTableColumn)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if (
            not index.isValid()
            or self._layer is None
            or role
            not in (
                QtCore.Qt.ItemDataRole.DisplayRole,
                QtCore.Qt.ItemDataRole.EditRole,
            )
        ):
            return None
        column = RoiTableColumn(index.column())
        if column is RoiTableColumn.NAME:
            return self._name(index.row())
        return str(self._layer.shape_type[index.row()])

    def headerData(
        self,
        section: int,
        orientation,
        role=QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if (
            orientation == QtCore.Qt.Orientation.Horizontal
            and role == QtCore.Qt.ItemDataRole.DisplayRole
        ):
            return RoiTableColumn(section).header
        return super().headerData(section, orientation, role)

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and RoiTableColumn(index.column()) is RoiTableColumn.NAME:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole) -> bool:
        if (
            role != QtCore.Qt.ItemDataRole.EditRole
            or not index.isValid()
            or self._layer is None
            or RoiTableColumn(index.column()) is not RoiTableColumn.NAME
        ):
            return False
        names = self.column_values(RoiTableColumn.NAME)
        names[index.row()] = str(value)
        features = self._layer.features.copy()
        features[RoiTableColumn.NAME.header] = names
        self._layer.features = features
        return True

    def column_values(self, column: RoiTableColumn) -> list[str]:
        """Project one complete column only for explicit bulk callers."""

        if self._layer is None:
            return []
        if column is RoiTableColumn.NAME:
            features = self._layer.features
            if RoiTableColumn.NAME.header in features.columns:
                return [
                    str(value)
                    for value in features[RoiTableColumn.NAME.header].tolist()
                ]
            return [
                f"ROI-{index:>04}" for index in range(self.rowCount())
            ]
        return [str(value) for value in self._layer.shape_type]

    def _name(self, row: int) -> str:
        if self._layer is None:
            raise RuntimeError("ROI table model has no native Shapes layer.")
        features = self._layer.features
        if RoiTableColumn.NAME.header in features.columns:
            return str(features[RoiTableColumn.NAME.header].iat[row])
        return f"ROI-{row:>04}"


class QRoiListWidget(QtW.QTableView):
    selected = QtCore.Signal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._roi_model = QRoiTableModel(self)
        self.setModel(self._roi_model)
        horizontal_header = self.horizontalHeader()
        horizontal_header.setFixedHeight(18)
        horizontal_header.setSectionResizeMode(
            RoiTableColumn.NAME,
            QtW.QHeaderView.ResizeMode.Stretch,
        )
        horizontal_header.setSectionResizeMode(
            RoiTableColumn.SHAPE_TYPE,
            QtW.QHeaderView.ResizeMode.Interactive,
        )
        horizontal_header.resizeSection(RoiTableColumn.SHAPE_TYPE, 96)
        self.verticalHeader().setSectionResizeMode(QtW.QHeaderView.ResizeMode.Fixed)
        self.verticalHeader().setDefaultSectionSize(18)
        self.setSelectionBehavior(QtW.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtW.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.selectionModel().selectionChanged.connect(self._selection_changed)

    def bind_layer(self, layer: Shapes | None) -> None:
        self._roi_model.bind_layer(layer)

    def refresh(self) -> None:
        self._roi_model.refresh()

    def rowCount(self) -> int:
        return self._roi_model.rowCount()

    def select_rows(self, rows: Iterable[int]) -> None:
        selection_model = self.selectionModel()
        blocker = QtCore.QSignalBlocker(selection_model)
        try:
            selection_model.clearSelection()
            flags = (
                QtCore.QItemSelectionModel.SelectionFlag.Select
                | QtCore.QItemSelectionModel.SelectionFlag.Rows
            )
            valid_rows = [
                row for row in sorted(set(rows)) if 0 <= row < self.rowCount()
            ]
            selection = QtCore.QItemSelection()
            if valid_rows:
                span_start = valid_rows[0]
                span_stop = span_start
                for row in (*valid_rows[1:], None):
                    if row is not None and row == span_stop + 1:
                        span_stop = row
                        continue
                    selection.select(
                        self.model().index(span_start, RoiTableColumn.NAME),
                        self.model().index(
                            span_stop,
                            RoiTableColumn.SHAPE_TYPE,
                        ),
                    )
                    if row is not None:
                        span_start = span_stop = row
                selection_model.select(selection, flags)
            if valid_rows:
                self.scrollTo(
                    self.model().index(valid_rows[0], RoiTableColumn.NAME),
                    QtW.QAbstractItemView.ScrollHint.EnsureVisible,
                )
        finally:
            del blocker

    def _selection_changed(self, _selected, _deselected):
        self.selected.emit(
            {index.row() for index in self.selectionModel().selectedRows()}
        )


class QRoiManager(QtW.QWidget):
    """Fiji-style manager bound directly to an ordinary native Shapes layer."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._layer: Shapes | None = None
        self._layer_connections: list[tuple[object, object]] = []
        self._visible_style: tuple[np.ndarray, np.ndarray] | None = None
        self._applying_hidden_style = False
        self._data_mutating = False
        self._viewer_selection_emitter = self._viewer.layers.selection.events.active
        self._viewer_selection_connected = False
        self._show_all = True
        self.setAcceptDrops(True)

        layout = QtW.QVBoxLayout(self)
        self._layer_name = QtW.QLabel("No Shapes layer selected", self)
        layout.addWidget(self._layer_name)
        content = QtW.QWidget(self)
        content_layout = QtW.QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content, 1)

        self._btns = QRoiManagerButtons(self)
        self._roilist = QRoiListWidget(self)
        size_policy = QtW.QSizePolicy.Policy
        alignment = QtCore.Qt.AlignmentFlag
        self._btns.setSizePolicy(size_policy.Fixed, size_policy.Preferred)
        self._roilist.setSizePolicy(size_policy.Expanding, size_policy.Expanding)
        content_layout.addWidget(self._roilist, 1)
        content_layout.addWidget(
            self._btns, 0, alignment.AlignTop | alignment.AlignRight
        )

        self._btns._add_roi_btn.clicked.connect(lambda _=False: self.register())
        self._btns._remove_roi_btn.clicked.connect(self._remove_button_clicked)
        self._btns._specify_btn.clicked.connect(self.specify_roi)
        self._btns._load_roiset_btn.clicked.connect(self.load_roiset)
        self._btns._save_roiset_btn.clicked.connect(self.save_roiset)
        self._btns._new_set_btn.clicked.connect(self.new_layer)
        self._btns._text_feature_name.currentIndexChanged.connect(
            self.set_text_feature_name
        )
        self._btns._text_font_size.valueChanged.connect(self.set_text_font_size)
        self._btns._show_all_checkbox.toggled.connect(self.set_show_all)
        self._roilist.selected.connect(self._roi_selected)

        self._connect_viewer_events()
        self._bind_active_layer()

    @classmethod
    def get_or_create(cls):
        """Get the RoiManager from the viewer or create a new one."""
        viewer = napari.current_viewer()
        if viewer is None:
            raise RuntimeError("No active viewer found.")
        for inner in viewer.window.dock_widgets.values():
            if isinstance(inner, cls):
                return inner
        return cls(viewer)

    def connect_layer(self, layer: Shapes) -> None:
        """Bind this manager directly to *layer*, without copying its state."""
        layer = _native_object(layer)
        if not isinstance(layer, Shapes):
            raise TypeError("ROI Manager can only bind a native napari Shapes layer")
        if layer is self._layer:
            return
        self._disconnect_layer()
        self._layer = layer
        for emitter, callback in (
            (layer.events.data, self._on_layer_data),
            (layer.events.features, self._on_layer_features),
            (layer.events.highlight, self._on_layer_highlight),
            (layer.events.name, self._on_layer_name),
            (layer.events.edge_color, self._on_layer_edge_color),
            (layer.events.face_color, self._on_layer_face_color),
        ):
            emitter.connect(callback)
            self._layer_connections.append((emitter, callback))
        if not self._show_all:
            self._hide_unselected_presentation()
        self._refresh_from_layer()

    def _disconnect_layer(self) -> None:
        self._restore_presentation()
        for emitter, callback in self._layer_connections:
            emitter.disconnect(callback)
        self._layer_connections.clear()
        self._layer = None
        self._roilist.bind_layer(None)
        self._layer_name.setText("No Shapes layer selected")
        self._refresh_text_features()
        self._update_controls()

    def _on_active_layer_changed(self, event) -> None:
        active = _native_object(event.value)
        if isinstance(active, Shapes):
            self.connect_layer(active)
        else:
            self._disconnect_layer()

    def _connect_viewer_events(self) -> None:
        if self._viewer_selection_connected:
            return
        self._viewer_selection_emitter.connect(self._on_active_layer_changed)
        self._viewer_selection_connected = True

    def _disconnect_viewer_events(self) -> None:
        if not self._viewer_selection_connected:
            return
        self._viewer_selection_emitter.disconnect(self._on_active_layer_changed)
        self._viewer_selection_connected = False

    def _bind_active_layer(self) -> None:
        active = _native_object(self._viewer.layers.selection.active)
        if isinstance(active, Shapes):
            self.connect_layer(active)
        else:
            self._disconnect_layer()

    def _on_layer_data(self, event) -> None:
        action = event.action
        if action in (
            ActionType.ADDING,
            ActionType.REMOVING,
            ActionType.CHANGING,
        ):
            self._data_mutating = True
        if self._visible_style is not None:
            if action is ActionType.ADDED:
                self._extend_hidden_style()
            elif action is ActionType.REMOVED:
                self._remove_from_hidden_style(event.data_indices)
        if action in (
            ActionType.ADDED,
            ActionType.REMOVED,
            ActionType.CHANGED,
        ):
            self._data_mutating = False
        if action not in (ActionType.ADDING, ActionType.REMOVING):
            self._refresh_from_layer()

    def _on_layer_features(self, _event) -> None:
        self._refresh_from_layer()

    def _on_layer_highlight(self, _event) -> None:
        if self._layer is not None:
            self._roilist.select_rows(self._layer.selected_data)

    def _on_layer_name(self, _event) -> None:
        if self._layer is not None:
            self._layer_name.setText(self._layer.name)

    def _on_layer_edge_color(self, _event) -> None:
        if (
            self._layer is None
            or self._visible_style is None
            or self._applying_hidden_style
            or self._data_mutating
        ):
            return
        saved_edge, saved_face = self._visible_style
        edge_color = np.array(self._layer.edge_color, copy=True)
        if len(edge_color) == len(saved_edge):
            edge_color[..., -1] = saved_edge[..., -1]
        self._visible_style = (
            edge_color,
            saved_face,
        )
        self._apply_hidden_style()

    def _on_layer_face_color(self, _event) -> None:
        if (
            self._layer is None
            or self._visible_style is None
            or self._applying_hidden_style
            or self._data_mutating
        ):
            return
        saved_edge, saved_face = self._visible_style
        face_color = np.array(self._layer.face_color, copy=True)
        if len(face_color) == len(saved_face):
            face_color[..., -1] = saved_face[..., -1]
        self._visible_style = (
            saved_edge,
            face_color,
        )
        self._apply_hidden_style()

    def _refresh_from_layer(self) -> None:
        layer = self._layer
        if layer is None:
            self._disconnect_layer()
            return
        self._layer_name.setText(layer.name)
        self._roilist.bind_layer(layer)
        self._roilist.select_rows(layer.selected_data)
        self._refresh_text_features()
        self._update_controls()

    def _roi_names(self) -> list[str]:
        layer = self._layer
        if layer is None:
            return []
        features = layer.features
        if "name" in features.columns:
            return [str(value) for value in features["name"].tolist()]
        return [f"ROI-{index:>04}" for index in range(len(layer.data))]

    def _refresh_text_features(self) -> None:
        combo = self._btns._text_feature_name
        current = combo.currentText()
        blocker = QtCore.QSignalBlocker(combo)
        try:
            combo.clear()
            if self._layer is not None:
                columns = [str(column) for column in self._layer.features.columns]
                combo.addItems(columns)
                if current in columns:
                    combo.setCurrentText(current)
        finally:
            del blocker

    def _update_controls(self) -> None:
        self._btns.set_layer_available(self._layer is not None)
        self._btns._load_roiset_btn.setEnabled(True)
        self._btns._new_set_btn.setEnabled(True)

    def new_layer(self, _checked: bool = False, *, ndim: int | None = None) -> Shapes:
        """Explicitly create and bind a new native Shapes ROI set."""
        if ndim is None:
            ndim = self._viewer.dims.ndim
        layer = _native_object(self._viewer.add_shapes(name="ROIs", ndim=ndim))
        self.connect_layer(layer)
        return layer

    def _require_layer(self, *, ndim: int | None = None) -> Shapes:
        if self._layer is None:
            return self.new_layer(ndim=ndim)
        return self._layer

    def add(
        self,
        data,
        shape_type: ShapeType = ShapeType.RECTANGLE,
    ):
        array = np.asarray(data)
        layer = self._require_layer(ndim=array.shape[-1])
        if array.shape[-1] != layer.ndim:
            raise ValueError(
                f"ROI dimensionality {array.shape[-1]} does not match layer ndim "
                f"{layer.ndim}"
            )
        before = len(layer.data)
        layer.add(data, shape_type=shape_type)
        layer.selected_data = set(range(before, len(layer.data)))
        return None

    def register(
        self,
        data=None,
        shape_type: ShapeType = ShapeType.RECTANGLE,
    ):
        """Register native geometry; supplied data is added exactly once."""
        if data is not None:
            self.add(data, shape_type=shape_type)
        layer = self._layer
        if layer is None:
            return None
        selected = set(layer.selected_data)
        if not selected and layer.data:
            selected = {len(layer.data) - 1}
        self._ensure_names(selected)
        if self._show_all:
            layer.selected_data = set()
        else:
            layer.selected_data = selected
        self._refresh_from_layer()
        return None

    def _ensure_names(self, indices: Iterable[int]) -> None:
        layer = self._layer
        if layer is None or not layer.data:
            return
        names = self._roi_names()
        for index in indices:
            if 0 <= index < len(names) and not names[index]:
                names[index] = f"ROI-{index:>04}"
        features = layer.features.copy()
        features["name"] = names
        layer.features = features

    def select(self, indices=()):
        layer = self._layer
        if layer is None:
            return
        if not isinstance(indices, Iterable):
            indices = {indices}
        layer.selected_data = {
            int(index) for index in indices if 0 <= int(index) < len(layer.data)
        }

    def remove(self, indices=None):
        layer = self._layer
        if layer is None:
            return
        if indices is not None:
            self.select(indices)
        layer.remove_selected()

    def specify_roi(self):
        layer = self._require_layer()
        from openhcs.napari_roi_manager.widgets._dialogs import QSpecifyDialog

        point = np.asarray(layer.world_to_data(self._viewer.dims.point), dtype=float)
        leading = tuple(point[:-2]) if layer.ndim > 2 else ()
        dialog = QSpecifyDialog(layer, leading, self)
        dialog.exec()
        self._refresh_from_layer()

    def _remove_button_clicked(self):
        self.remove()

    def _roi_selected(self, indices: set[int]) -> None:
        if self._layer is not None:
            self._layer.selected_data = {
                index for index in indices if index < len(self._layer.data)
            }

    def set_text_feature_name(self, _index: int):
        layer = self._layer
        feature_name = self._btns._text_feature_name.currentText()
        if layer is not None and feature_name:
            layer.text.string = "{" + feature_name + "}"
            layer.text.visible = True

    def set_text_font_size(self, size: int):
        if self._layer is not None:
            self._layer.text.size = size

    def set_show_all(self, show_all: bool):
        self._show_all = bool(show_all)
        checkbox = self._btns._show_all_checkbox
        if checkbox.isChecked() != self._show_all:
            blocker = QtCore.QSignalBlocker(checkbox)
            try:
                checkbox.setChecked(self._show_all)
            finally:
                del blocker
        if self._layer is None:
            return
        if self._show_all:
            self._restore_presentation()
        else:
            self._hide_unselected_presentation()

    def _hide_unselected_presentation(self) -> None:
        if self._layer is None or self._visible_style is not None:
            return
        edge_color = np.array(self._layer.edge_color, copy=True)
        face_color = np.array(self._layer.face_color, copy=True)
        self._visible_style = (edge_color, face_color)
        self._apply_hidden_style()
        self._layer.events.highlight()

    def _apply_hidden_style(self) -> None:
        if self._layer is None or self._visible_style is None:
            return
        edge_color = np.array(self._layer.edge_color, copy=True)
        face_color = np.array(self._layer.face_color, copy=True)
        edge_color[..., -1] = 0.0
        face_color[..., -1] = 0.0
        self._applying_hidden_style = True
        try:
            self._layer.edge_color = edge_color
            self._layer.face_color = face_color
        finally:
            self._applying_hidden_style = False

    def _extend_hidden_style(self) -> None:
        if self._layer is None or self._visible_style is None:
            return
        saved_edge, saved_face = self._visible_style
        current_edge = np.asarray(self._layer.edge_color)
        current_face = np.asarray(self._layer.face_color)
        if len(current_edge) > len(saved_edge):
            saved_edge = np.concatenate(
                [saved_edge, current_edge[len(saved_edge) :]], axis=0
            )
            saved_face = np.concatenate(
                [saved_face, current_face[len(saved_face) :]], axis=0
            )
            self._visible_style = (saved_edge, saved_face)
        self._apply_hidden_style()

    def _remove_from_hidden_style(self, indices: Iterable[int]) -> None:
        if self._layer is None or self._visible_style is None:
            return
        saved_edge, saved_face = self._visible_style
        normalized = sorted(
            {index if index >= 0 else len(saved_edge) + index for index in indices}
        )
        if normalized:
            saved_edge = np.delete(saved_edge, normalized, axis=0)
            saved_face = np.delete(saved_face, normalized, axis=0)
            self._visible_style = (saved_edge, saved_face)
        self._apply_hidden_style()

    def _restore_presentation(self) -> None:
        if self._layer is not None and self._visible_style is not None:
            edge_color, face_color = self._visible_style
            self._applying_hidden_style = True
            try:
                self._layer.edge_color = edge_color
                self._layer.face_color = face_color
            finally:
                self._applying_hidden_style = False
        self._visible_style = None

    def _roi_data(self) -> RoiData:
        layer = self._layer
        if layer is None:
            return RoiData()
        names = self._roi_names() if layer.data else []
        features = {
            str(column): layer.features[column].tolist()
            for column in layer.features.columns
        }
        return RoiData(
            data=[np.asarray(data) for data in layer.data],
            shape_type=[ShapeType(value) for value in layer.shape_type],
            names=names,
            features=features,
        )

    def _clear_layer(self) -> None:
        layer = self._layer
        if layer is None or not layer.data:
            return
        layer.selected_data = set(range(len(layer.data)))
        layer.remove_selected()

    def _append_roi_data(self, rois: RoiData) -> None:
        if not rois.data:
            return
        ndim = np.asarray(rois.data[0]).shape[-1]
        layer = self._require_layer(ndim=ndim)
        if layer.ndim != ndim:
            raise ValueError(
                f"ROI dimensionality {ndim} does not match layer ndim {layer.ndim}"
            )
        start = len(layer.data)
        existing_names = self._roi_names()
        existing_features = {
            str(column): layer.features[column].tolist()
            for column in layer.features.columns
        }
        existing_features["name"] = existing_names
        layer.add(rois.data, shape_type=rois.shape_type)
        incoming_names = rois.names or [
            f"ROI-{index:>04}" for index in range(start, len(layer.data))
        ]
        incoming_features = {
            column: list(values) for column, values in (rois.features or {}).items()
        }
        incoming_features.setdefault("name", list(incoming_names))
        incoming_count = len(rois.data)
        columns = tuple(dict.fromkeys((*existing_features, *incoming_features)))
        layer.features = {
            column: existing_features.get(column, [None] * start)
            + incoming_features.get(column, [None] * incoming_count)
            for column in columns
        }

    def load_roiset(self, _checked: bool = False, *, path=None, append: bool = True):
        if path is None:
            file = QtW.QFileDialog.getOpenFileName(
                self,
                "Open ROIs",
                filter="JSON (*.json *.txt);;ImageJ ROI (*.roi *.zip);;All Files (*)",
            )
            if not file or not file[0]:
                return
            path = file[0]
            if self._layer is not None and self._layer.data:
                result = QCustomDialog.construct(
                    title="Append or Replace",
                    message=(
                        "ROIs are already registered. Do you want to append the "
                        "incoming ROIs or replace the existing ones?"
                    ),
                    choices=["Append", "Replace", "Cancel"],
                    parent=self,
                ).exec_dialog()
                if result == "Replace":
                    append = False
                elif result is None or result == "Cancel":
                    return
        path = Path(path)
        if path.suffix in (".zip", ".roi"):
            self.read_ij_roi(path, append=append)
        else:
            with path.open() as stream:
                rois = RoiData.from_json_dict(json.load(stream))
            if not append:
                self._clear_layer()
            self._append_roi_data(rois)
        self._refresh_from_layer()

    def save_roiset(self, _checked: bool = False, *, path=None):
        if path is None:
            file = QtW.QFileDialog.getSaveFileName(
                self,
                "Save ROIs",
                filter="JSON (*.json *.txt);;ImageJ ROI (*.zip);;All Files (*)",
            )
            if not file or not file[0]:
                return
            path = file[0]
        path = Path(path)
        if path.suffix == ".zip":
            self.write_ij_roi(path)
        else:
            with path.open("w") as stream:
                json.dump(self._roi_data().to_json_dict(), stream)

    def read_ij_roi(self, path, append: bool = True):
        ijrois = roiread(path)
        if not isinstance(ijrois, list):
            ijrois = [ijrois]
        converted = [roi_to_shape(roi) for roi in ijrois]
        converted = [shape for shape in converted if shape is not None]
        if not append:
            self._clear_layer()
        layer = self._require_layer()
        n_leading = layer.ndim - 2
        point = np.asarray(layer.world_to_data(self._viewer.dims.point), dtype=float)
        current_leading = point[:-2]
        data: list[np.ndarray] = []
        shape_types: list[ShapeType] = []
        names: list[str] = []
        for index, shape in enumerate(converted):
            plane = np.asarray(shape.data)
            if plane.ndim == 3 and len(plane) == 1:
                plane = plane[0]
            if n_leading:
                leading = np.asarray(current_leading, dtype=plane.dtype).copy()
                source_leading = np.asarray(shape.multidim[-n_leading:])
                leading[-len(source_leading) :] = source_leading
                prefix = np.broadcast_to(leading, (len(plane), n_leading))
                coords = np.concatenate([prefix, plane[:, -2:]], axis=1)
            else:
                coords = plane[:, -2:]
            data.append(coords)
            shape_types.append(shape.shape_type)
            names.append(shape.name or f"ROI-{len(layer.data) + index:>04}")
        self._append_roi_data(RoiData(data, shape_types, names))
        return []

    def write_ij_roi(self, path):
        path = Path(path)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if path.exists():
            path.unlink()
        layer = self._layer
        if layer is None:
            roiwrite(path, [])
            return
        names = self._roi_names()
        rois = []
        for index, (data, shape_type) in enumerate(
            zip(layer.data, layer.shape_type, strict=True)
        ):
            array = np.asarray(data)
            multidim = tuple(int(round(value)) for value in array[0, :-2])
            rois.append(
                shape_to_roi(
                    RoiTuple(
                        data=array[:, -2:],
                        shape_type=ShapeType(shape_type),
                        name=names[index],
                        multidim=multidim,
                    )
                )
            )
        roiwrite(path, rois)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        for url in event.mimeData().urls():
            self.load_roiset(path=Path(url.toLocalFile()), append=True)
        event.accept()

    def closeEvent(self, event):
        self._disconnect_layer()
        self._disconnect_viewer_events()
        super().closeEvent(event)

    def hideEvent(self, event):
        self._disconnect_layer()
        self._disconnect_viewer_events()
        super().hideEvent(event)

    def showEvent(self, event):
        self._connect_viewer_events()
        self._bind_active_layer()
        super().showEvent(event)
