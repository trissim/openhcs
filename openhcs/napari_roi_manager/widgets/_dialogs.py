from __future__ import annotations

from textwrap import wrap

import numpy as np
from napari.layers import Shapes
from qtpy import QtCore, QtGui
from qtpy import QtWidgets as QtW


class QCustomDialog(QtW.QDialog):
    def __init__(self, parent: QtW.QWidget | None = None):
        super().__init__(parent)

        layout = QtW.QVBoxLayout(self)
        self._message = QtW.QLabel(self)
        layout.addWidget(self._message)

        self._buttons = QtW.QWidget(self)
        _btn_layout = QtW.QHBoxLayout(self._buttons)
        _btn_layout.setContentsMargins(0, 0, 0, 0)
        _btn_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._buttons)
        self._last_clicked = None

    def exec_dialog(self) -> str | None:
        if self.exec():
            return self._last_clicked
        else:
            return None

    @classmethod
    def construct(
        cls,
        title: str,
        message: str,
        choices: list[str],
        parent: QtW.QWidget | None = None,
    ) -> QCustomDialog:
        self = cls(parent)
        self.setWindowTitle(title)
        self._message.setText("\n".join(wrap(message)))
        for choice in choices:
            btn = QtW.QPushButton(choice, self._buttons)
            self._buttons.layout().addWidget(btn)
            btn.clicked.connect(self._make_callback(choice))
        return self

    def _make_callback(self, choice: str):
        def callback():
            self._last_clicked = choice
            self.accept()

        return callback


def _labeled(text: str, widget: QtW.QWidget) -> QtW.QWidget:
    out = QtW.QWidget()
    layout = QtW.QHBoxLayout()
    layout.addWidget(QtW.QLabel(text))
    layout.addWidget(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    out.setLayout(layout)
    return out


class QSpecifyDialog(QtW.QDialog):
    """Create or edit a rectangle in an ordinary native Shapes layer."""

    def __init__(
        self,
        layer: Shapes,
        leading_position: tuple[float, ...] = (),
        parent: QtW.QWidget | None = None,
    ):
        super().__init__(parent)

        self.layer = layer
        self._leading_position = np.asarray(leading_position, dtype=float)
        selected = sorted(layer.selected_data)
        self._shape_index = (
            selected[0]
            if len(selected) == 1 and layer.shape_type[selected[0]] == "rectangle"
            else None
        )
        layout = QtW.QVBoxLayout(self)
        self.width_input = QtW.QLineEdit("256", self)
        self.width_input.setValidator(QtGui.QDoubleValidator())
        self.height_input = QtW.QLineEdit("256", self)
        self.height_input.setValidator(QtGui.QDoubleValidator())
        self.x_input = QtW.QLineEdit("128", self)
        self.x_input.setValidator(QtGui.QDoubleValidator())
        self.y_input = QtW.QLineEdit("128", self)
        self.y_input.setValidator(QtGui.QDoubleValidator())
        self.multiply_by_scale = QtW.QCheckBox("Multiply by scale", self)
        self.run_button = QtW.QPushButton("OK", self)

        layout.addWidget(_labeled("Width", self.width_input))
        layout.addWidget(_labeled("Height", self.height_input))
        layout.addWidget(_labeled("X coordinate", self.x_input))
        layout.addWidget(_labeled("Y coordinate", self.y_input))
        layout.addWidget(self.multiply_by_scale)
        layout.addWidget(self.run_button)

        self.width_input.textChanged.connect(self._on_value_changed)
        self.height_input.textChanged.connect(self._on_value_changed)
        self.x_input.textChanged.connect(self._on_value_changed)
        self.y_input.textChanged.connect(self._on_value_changed)
        self.multiply_by_scale.stateChanged.connect(self._on_value_changed)

        self.run_button.clicked.connect(self.accept)

        if self._shape_index is not None:
            data = np.asarray(layer.data[self._shape_index])
            y0, x0 = data[:, -2:].min(axis=0)
            y1, x1 = data[:, -2:].max(axis=0)
            self.width_input.setText(str(x1 - x0))
            self.height_input.setText(str(y1 - y0))
            self.x_input.setText(str(x0))
            self.y_input.setText(str(y0))
            self._leading_position = data[0, :-2]

        self._on_value_changed()

    def _on_value_changed(self):
        yscale, xscale = (
            self.layer.scale[-2:] if self.multiply_by_scale.isChecked() else (1, 1)
        )
        try:
            width = float(self.width_input.text()) * xscale
            height = float(self.height_input.text()) * yscale
            x = float(self.x_input.text()) * xscale
            y = float(self.y_input.text()) * yscale
        except ValueError:
            return
        plane_data = np.asarray(
            [[y, x], [y + height, x], [y + height, x + width], [y, x + width]],
            dtype=float,
        )
        if self.layer.ndim > 2:
            leading = np.broadcast_to(
                self._leading_position, (len(plane_data), self.layer.ndim - 2)
            )
            new_data = np.concatenate([leading, plane_data], axis=1)
        else:
            new_data = plane_data
        if self._shape_index is None:
            self.layer.add(new_data, shape_type="rectangle")
            self._shape_index = len(self.layer.data) - 1
        else:
            data = list(self.layer.data)
            data[self._shape_index] = new_data
            self.layer.data = data
        self.layer.selected_data = {self._shape_index}
