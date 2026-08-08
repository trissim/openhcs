"""Rendering regression for scoped Source Bindings preview tables."""

from __future__ import annotations

from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QPalette

from openhcs.core.config import SourceBindingsConfig, StepSourceBindingsConfig
from openhcs.pyqt_gui.widgets.source_bindings_editor import SourceBindingsEditorWidget
from pyqt_reactive.theming import ColorScheme, PaletteManager
from pyqt_reactive.widgets.shared.scope_color_utils import get_scope_color_scheme
from pyqt_reactive.widgets.shared.scoped_table_widget import ScopedTableWidget


def test_source_bindings_pipeline_sources_preview_renders_palette_surface(qapp) -> None:
    color_scheme = ColorScheme()
    original_palette = qapp.palette()
    qapp.setPalette(PaletteManager(color_scheme).create_palette())
    widget = SourceBindingsEditorWidget.from_bindings(StepSourceBindingsConfig())
    try:
        widget.set_preview_context(source_bindings=SourceBindingsConfig())
        widget.setStyleSheet(
            color_scheme.styles.generate_config_window_style()
        )
        widget.set_scope_color_scheme(
            get_scope_color_scheme("plate::step_0", step_index=0)
        )
        widget.resize(430, 900)
        widget.show()
        for _ in range(8):
            qapp.processEvents()

        pipeline_sources_table = widget.findChildren(ScopedTableWidget)[0]
        assert pipeline_sources_table.rowCount() == 2
        assert pipeline_sources_table.item(0, 0).text() == "image-plane sources"
        assert pipeline_sources_table.item(1, 0).text() == "imported metadata tables"

        viewport = pipeline_sources_table.viewport()
        surface_position = viewport.mapTo(
            pipeline_sources_table,
            QPoint(viewport.width() - 8, viewport.height() - 8),
        )
        rendered = pipeline_sources_table.grab().toImage()

        assert rendered.pixelColor(surface_position) == (
            pipeline_sources_table.palette().color(QPalette.ColorRole.Base)
        )
    finally:
        widget.close()
        widget.deleteLater()
        qapp.setPalette(original_palette)
