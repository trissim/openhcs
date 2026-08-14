"""Function Selector leaves generic column filtering with its table browser."""

from concurrent.futures import Future

from PyQt6.QtTest import QTest
from pyqt_reactive.widgets.shared.abstract_table_browser import ColumnPresentation

from openhcs.agent.dto.functions import FunctionCatalogEntry, catalog_page
from openhcs.processing.custom_functions.signals import CustomFunctionSignals
from openhcs.pyqt_gui.dialogs import function_selector_dialog as selector_module
from openhcs.pyqt_gui.dialogs.function_selector_dialog import FunctionSelectorDialog


class _FunctionCatalog:
    def __init__(self) -> None:
        self.entries = (
            FunctionCatalogEntry(
                function_id="core_fn",
                import_path=f"{__name__}._core_fn",
                name="core_fn",
                module="package.core",
                library="core",
                signature="core_fn()",
                summary="Core function",
                backend_tags=("segmentation", "shared"),
            ),
            FunctionCatalogEntry(
                function_id="plugin_fn",
                import_path=f"{__name__}._plugin_fn",
                name="plugin_fn",
                module="package.plugin",
                library="plugin",
                signature="plugin_fn()",
                summary="Plugin function",
                backend_tags=("measurement", "shared"),
            ),
        )

    def catalog(self, *, compact_signatures: bool = True):
        assert compact_signatures
        return catalog_page(
            items=self.entries,
            total=len(self.entries),
            limit=len(self.entries),
            query=None,
            library=None,
        )

    def prepare(self, *, compact_signatures: bool = True):
        future = Future()
        future.set_result(self.catalog(compact_signatures=compact_signatures))
        return future

    def invalidate(self) -> None:
        pass

    def import_selected_callable(self, function_id: str):
        return _core_fn if function_id == "core_fn" else _plugin_fn


def _core_fn():
    return None


def _plugin_fn():
    return None


def test_function_selector_has_no_column_filter_plumbing(qapp, monkeypatch) -> None:
    signals = CustomFunctionSignals()
    monkeypatch.setattr(selector_module, "custom_function_signals", signals)
    dialog = FunctionSelectorDialog(_FunctionCatalog())
    dialog.show()
    qapp.processEvents()

    try:
        table_browser = dialog.function_table_browser
        panel = table_browser.column_filter_panel
        presentation = table_browser.column_presentation

        assert not hasattr(dialog, "column_filter_panel")
        assert not hasattr(dialog, "_build_column_filters")
        assert not hasattr(dialog, "_on_column_filters_changed")
        assert tuple(panel.column_filters) == (
            "library",
            "backend_tags",
        )
        assert panel.column_filters["backend_tags"].unique_values == [
            "measurement",
            "segmentation",
            "shared",
        ]

        assert table_browser.set_column_filter_selection("library", ("core",))
        assert tuple(table_browser.filtered_items) == ("core_fn",)

        dialog._update_filtered_view(
            {"plugin_fn": dialog.all_functions_metadata["plugin_fn"]},
            "filtered by module",
        )
        assert table_browser.filtered_items == {}

        presentation.set_preference(
            ColumnPresentation(
                ordered_keys=(
                    "backend_tags",
                    "library",
                    "name",
                    "module",
                    "summary",
                ),
                hidden_keys=frozenset({"library"}),
            )
        )
        QTest.qWait(10)
        qapp.processEvents()

        assert tuple(panel.column_filters) == (
            "backend_tags",
            "library",
        )
        assert table_browser.table_widget.isColumnHidden(2)
        assert panel.column_filters["library"].isHidden()
        assert panel.hidden_active_label.text() == "1 hidden active filter"
    finally:
        signals.functions_changed.disconnect(dialog._on_functions_changed)
        dialog.close()
        dialog.deleteLater()


def test_function_selector_renders_endpoint_catalog_entry_directly(qapp) -> None:
    entry = _FunctionCatalog().entries[0]
    browser = selector_module.FunctionTableBrowser()
    browser.set_items({entry.function_id: entry})

    assert browser.extract_row_data(entry) == [
        "core_fn",
        "package.core",
        "core",
        "segmentation, shared",
        "Core function",
    ]
