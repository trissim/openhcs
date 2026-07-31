"""Stable geometry contracts for the embedded main-window panes."""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSplitter, QWidget

from openhcs.pyqt_gui.services.main_window_workflows import (
    MainWindowEmbeddedWidgets,
)


@pytest.mark.parametrize(
    "show_method_name",
    ("show_plate_manager", "show_pipeline_editor", "show_zmq_manager"),
)
def test_showing_embedded_widget_preserves_user_splitter_sizes(
    qapp,
    show_method_name: str,
) -> None:
    """Focus/show routes must not replace the user's current pane proportions."""

    main_splitter = QSplitter(Qt.Orientation.Horizontal)
    left_splitter = QSplitter(Qt.Orientation.Vertical)
    plate_manager = QWidget()
    zmq_manager = QWidget()
    pipeline_editor = QWidget()
    left_splitter.addWidget(plate_manager)
    left_splitter.addWidget(zmq_manager)
    main_splitter.addWidget(left_splitter)
    main_splitter.addWidget(pipeline_editor)
    main_splitter.resize(900, 600)
    main_splitter.show()
    qapp.processEvents()
    main_splitter.setSizes([570, 330])
    left_splitter.setSizes([410, 190])
    qapp.processEvents()

    embedded = MainWindowEmbeddedWidgets(
        plate_manager=plate_manager,
        pipeline_editor=pipeline_editor,
        zmq_manager=zmq_manager,
        left_splitter=left_splitter,
        main_splitter=main_splitter,
    )
    main_sizes_before = main_splitter.sizes()
    left_sizes_before = left_splitter.sizes()

    getattr(embedded, show_method_name)()
    qapp.processEvents()

    assert main_splitter.sizes() == main_sizes_before
    assert left_splitter.sizes() == left_sizes_before
    main_splitter.close()
