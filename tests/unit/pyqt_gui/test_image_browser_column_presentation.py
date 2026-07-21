"""Image Browser uses the generic table owner without filter UI plumbing."""

from PyQt6.QtTest import QTest
from pyqt_reactive.widgets.shared.abstract_table_browser import ColumnPresentation

from openhcs.pyqt_gui.widgets.image_browser import ImageBrowserItem, ImageBrowserWidget


def _settle(qapp) -> None:
    for _ in range(5):
        qapp.processEvents()
        QTest.qWait(10)


class _PlateSelectionRecorder:
    def __init__(self) -> None:
        self.selected: set[str] = set()

    def select_wells(self, well_ids: set[str], *, emit_signal: bool) -> None:
        assert not emit_signal
        self.selected = well_ids


def test_image_browser_delegates_columns_filters_and_well_sync_to_table_owner(
    qapp,
    monkeypatch,
) -> None:
    browser = ImageBrowserWidget()
    browser.resize(1200, 800)
    browser.show()
    monkeypatch.setattr(
        browser.metadata_display_resolver,
        "_resolve_display_value",
        lambda _key, value: f"{value} | label",
    )
    browser.file_items = {
        "a.tif": ImageBrowserItem(
            key="a.tif",
            metadata={"filename": "a.tif", "well": "A01", "channel": "W1"},
        ),
        "b.tif": ImageBrowserItem(
            key="b.tif",
            metadata={"filename": "b.tif", "well": "A02", "channel": "W2"},
        ),
    }
    browser.metadata_keys = ["well", "channel"]
    browser.image_table_browser.set_metadata_keys(browser.metadata_keys)
    browser.image_table_browser.set_items(browser.file_items)
    _settle(qapp)

    try:
        table_browser = browser.image_table_browser
        panel = table_browser.column_filter_panel
        presentation = table_browser.column_presentation

        assert not hasattr(browser, "column_filter_panel")
        assert not hasattr(browser, "column_presentation")
        assert not hasattr(browser.filter_controller, "build_column_filters")
        assert tuple(panel.column_filters) == ("well", "channel")
        assert panel.column_filters["well"].unique_values == [
            "A01 | label",
            "A02 | label",
        ]
        assert table_browser.table_widget.item(0, 1).text() == "A01 | label"

        plate = _PlateSelectionRecorder()
        browser.plate_view_widget = plate
        assert table_browser.set_column_filter_selection(
            "well", ("A01 | label",)
        )
        assert tuple(table_browser.filtered_items) == ("a.tif",)
        assert plate.selected == {"A01"}

        browser._on_wells_selected({"A02"})
        assert tuple(table_browser.filtered_items) == ("b.tif",)
        assert table_browser.column_filter_selection("well") == frozenset(
            {"A02 | label"}
        )
        assert browser.selected_wells == set()

        presentation.set_preference(
            ColumnPresentation(
                ordered_keys=("channel", "filename", "well"),
                hidden_keys=frozenset({"well"}),
            )
        )
        _settle(qapp)

        header = table_browser.table_widget.horizontalHeader()
        columns = presentation.columns
        assert tuple(
            columns[header.logicalIndex(index)].key
            for index in range(header.count())
        ) == ("channel", "filename", "well")
        assert table_browser.table_widget.isColumnHidden(1)
        assert panel.column_filters["well"].isHidden()
        assert panel.hidden_active_label.text() == "1 hidden active filter"
    finally:
        browser.close()
        browser.deleteLater()
