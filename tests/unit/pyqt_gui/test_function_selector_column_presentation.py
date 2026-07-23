"""Function Selector leaves generic column filtering with its table browser."""

from dataclasses import dataclass

from PyQt6.QtTest import QTest
from pyqt_reactive.widgets.shared.abstract_table_browser import ColumnPresentation

from openhcs.processing.custom_functions.signals import CustomFunctionSignals
from openhcs.pyqt_gui.dialogs import function_selector_dialog as selector_module
from openhcs.pyqt_gui.dialogs.function_selector_dialog import FunctionSelectorDialog


@dataclass(frozen=True)
class _FunctionMetadata:
    name: str
    module: str
    backend: str
    registry: str
    contract: str
    tags: tuple[str, ...]
    doc: str

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def func(self):
        return lambda: None

    def get_memory_type(self) -> str:
        return self.backend

    def get_registry_name(self) -> str:
        return self.registry


def test_function_selector_has_no_column_filter_plumbing(qapp, monkeypatch) -> None:
    signals = CustomFunctionSignals()
    monkeypatch.setattr(selector_module, "custom_function_signals", signals)
    prior_cache = FunctionSelectorDialog._metadata_cache
    FunctionSelectorDialog._metadata_cache = {
        "core_fn": _FunctionMetadata(
            name="core_fn",
            module="package.core",
            backend="cpu",
            registry="core",
            contract="FLEXIBLE",
            tags=("segmentation", "shared"),
            doc="Core function",
        ),
        "plugin_fn": _FunctionMetadata(
            name="plugin_fn",
            module="package.plugin",
            backend="gpu",
            registry="plugin",
            contract="PURE_2D",
            tags=("measurement", "shared"),
            doc="Plugin function",
        ),
    }
    dialog = FunctionSelectorDialog()
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
            "backend",
            "registry",
            "contract",
            "tags",
        )
        assert panel.column_filters["tags"].unique_values == [
            "measurement",
            "segmentation",
            "shared",
        ]

        assert table_browser.set_column_filter_selection("registry", ("Core",))
        assert tuple(table_browser.filtered_items) == ("core_fn",)

        dialog._update_filtered_view(
            {"plugin_fn": dialog.all_functions_metadata["plugin_fn"]},
            "filtered by module",
        )
        assert table_browser.filtered_items == {}

        presentation.set_preference(
            ColumnPresentation(
                ordered_keys=(
                    "tags",
                    "registry",
                    "backend",
                    "contract",
                    "name",
                    "module",
                    "doc",
                ),
                hidden_keys=frozenset({"registry"}),
            )
        )
        QTest.qWait(10)
        qapp.processEvents()

        assert tuple(panel.column_filters) == (
            "tags",
            "registry",
            "backend",
            "contract",
        )
        assert table_browser.table_widget.isColumnHidden(3)
        assert panel.column_filters["registry"].isHidden()
        assert panel.hidden_active_label.text() == "1 hidden active filter"
    finally:
        signals.functions_changed.disconnect(dialog._on_functions_changed)
        dialog.close()
        dialog.deleteLater()
        FunctionSelectorDialog._metadata_cache = prior_cache
