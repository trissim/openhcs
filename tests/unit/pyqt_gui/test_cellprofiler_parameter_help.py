"""Generic parameter-help behavior used by CellProfiler function panes."""

import ast
import inspect


def test_parameter_help_content_uses_parameter_window_not_docstring_mirror(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import openhcs  # noqa: F401 - activates source-checkout externals first
    from PyQt6.QtWidgets import QApplication
    from pyqt_reactive.windows.help_window_manager import (
        HELP_WINDOW_CONTENT_MARGIN,
        HELP_WINDOW_DIALOG_MARGIN,
        HELP_WINDOW_MIN_WIDTH,
        DocstringHelpWindow,
        HelpWindowManager,
        ParameterHelpWindow,
    )

    app = QApplication.instance() or QApplication([])
    HelpWindowManager._help_window = None

    try:
        HelpWindowManager.show_parameter_help(
            "source_filters",
            "Filters limiting the source universe before named bindings are resolved.",
            parent=None,
        )
        for _ in range(20):
            app.processEvents()

        window = HelpWindowManager._help_window
        assert isinstance(window, ParameterHelpWindow)
        assert not isinstance(window, DocstringHelpWindow)
        assert window.content.summary == "• source_filters"
        assert (
            window.content.description
            == "Filters limiting the source universe before named bindings are resolved."
        )
        assert window.layout().contentsMargins().left() == HELP_WINDOW_DIALOG_MARGIN
        content_layout = window.content_area.widget().layout()
        assert content_layout.contentsMargins().left() == HELP_WINDOW_CONTENT_MARGIN
        assert window.width() >= HELP_WINDOW_MIN_WIDTH
        assert window.minimumWidth() >= HELP_WINDOW_MIN_WIDTH
        assert window.content_area.minimumHeight() == window.content_area.maximumHeight()
        assert window.height() >= window.sizeHint().height()
    finally:
        if HelpWindowManager._help_window is not None:
            HelpWindowManager._help_window.close()
        HelpWindowManager._help_window = None


def test_callable_parameter_popup_uses_wrapper_owned_descriptions(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import openhcs  # noqa: F401 - activates source-checkout externals first
    from arraybridge.decorators import numpy
    from PyQt6.QtWidgets import QApplication
    from python_introspect import mark_enableable
    from pyqt_reactive.windows.help_window_manager import (
        HelpWindowManager,
        ParameterHelpWindow,
    )

    @numpy
    def slice_probe(image):
        """Return the input image.

        Args:
            image: Input image stack.
        """

        return image

    def enableable_probe(image, *, enabled: bool = True):
        """Return the input image when enabled.

        Args:
            image: Input image stack.
        """

        return image

    mark_enableable(enableable_probe)
    app = QApplication.instance() or QApplication([])
    HelpWindowManager._help_window = None

    try:
        cases = (
            (
                slice_probe,
                "slice_by_slice",
                ("numpy memory decorator", "Process 3D arrays slice-by-slice"),
            ),
            (
                enableable_probe,
                "enabled",
                (
                    "Run this callable or configuration when enabled; "
                    "skip it when disabled.",
                ),
            ),
        )
        for target, parameter_name, expected_fragments in cases:
            HelpWindowManager.show_parameter_help(
                parameter_name,
                f"Parameter: {parameter_name}",
                help_target=target,
                parent=None,
            )
            for _ in range(20):
                app.processEvents()

            window = HelpWindowManager._help_window
            assert isinstance(window, ParameterHelpWindow)
            assert window.content.summary == f"• {parameter_name}"
            assert window.content.description != f"Parameter: {parameter_name}"
            assert all(
                fragment in window.content.description
                for fragment in expected_fragments
            )
    finally:
        if HelpWindowManager._help_window is not None:
            HelpWindowManager._help_window.close()
        HelpWindowManager._help_window = None


def test_callable_popup_projection_has_no_parameter_name_dispatch() -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from pyqt_reactive.services.parameter_help_service import (
        parameter_description_from_target,
    )

    source = inspect.getsource(parameter_description_from_target)
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "docstring_info_for_target" in called_names
    assert "dataclass_type_for_target" not in called_names
    assert {"enabled", "slice_by_slice"}.isdisjoint(string_literals)


def test_dataclass_help_replaces_parameter_window_without_reusing_parameter_dialog(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import openhcs  # noqa: F401 - activates source-checkout externals first
    from PyQt6.QtWidgets import QApplication
    from openhcs.core.config import PathPlanningConfig
    from pyqt_reactive.windows.help_window_manager import (
        DocstringHelpWindow,
        HelpWindowManager,
        ParameterHelpWindow,
    )

    app = QApplication.instance() or QApplication([])
    HelpWindowManager._help_window = None

    try:
        HelpWindowManager.show_parameter_help(
            "source_filters",
            "Filters limiting the source universe before named bindings are resolved.",
            parent=None,
        )
        for _ in range(20):
            app.processEvents()

        parameter_window = HelpWindowManager._help_window
        assert isinstance(parameter_window, ParameterHelpWindow)

        HelpWindowManager.show_docstring_help(PathPlanningConfig, parent=None)
        for _ in range(20):
            app.processEvents()

        dataclass_window = HelpWindowManager._help_window
        assert isinstance(dataclass_window, DocstringHelpWindow)
        assert dataclass_window is not parameter_window
        assert dataclass_window.target is PathPlanningConfig
        assert "PathPlanningConfig(" not in dataclass_window.docstring_info.summary
    finally:
        if HelpWindowManager._help_window is not None:
            HelpWindowManager._help_window.close()
        HelpWindowManager._help_window = None


def test_help_context_routes_parameter_help_with_function_target(monkeypatch) -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from openhcs.processing.backends import cellprofiler
    from pyqt_reactive.widgets.shared.clickable_help_components import HelpContext
    from pyqt_reactive.windows.help_window_manager import HelpWindowManager

    calls = []

    def record_parameter_help(*args, **kwargs):
        calls.append(("parameter", args, kwargs))

    def record_docstring_help(*args, **kwargs):
        calls.append(("docstring", args, kwargs))

    monkeypatch.setattr(HelpWindowManager, "show_parameter_help", record_parameter_help)
    monkeypatch.setattr(HelpWindowManager, "show_docstring_help", record_docstring_help)

    shown = HelpContext(
        help_target=cellprofiler.measure_object_intensity,
        param_name="labels",
        param_description="Parameter: Labels",
    ).show_help(parent_widget=None)

    assert shown is True
    assert len(calls) == 1
    kind, args, kwargs = calls[0]
    assert kind == "parameter"
    assert args[:2] == ("labels", "Parameter: Labels")
    assert kwargs["help_target"] is cellprofiler.measure_object_intensity


def test_dataclass_help_uses_source_field_docs_instead_of_signature() -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from openhcs.core.config import LazyPathPlanningConfig, PathPlanningConfig
    from pyqt_reactive.windows.help_window_manager import (
        docstring_info_for_target,
        resolved_parameter_description,
    )

    docstring_info = docstring_info_for_target(LazyPathPlanningConfig)

    assert docstring_info.summary == (
        "Configuration for pipeline path planning and directory structure."
    )
    assert "PathPlanningConfig(" not in docstring_info.summary
    assert "output_dir_suffix" in docstring_info.parameters
    assert (
        docstring_info.parameters["output_dir_suffix"]
        == "Default suffix for general step output directories."
    )

    description = resolved_parameter_description(
        help_target=PathPlanningConfig,
        param_name="global_output_folder",
        widget_description="Parameter: Global Output Folder",
    )

    assert "Optional global output folder" in description
    assert description != "Parameter: Global Output Folder"


def test_nested_dataclass_form_uses_nested_dataclass_help_target(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    import openhcs  # noqa: F401 - activates source-checkout externals first
    from PyQt6.QtWidgets import QApplication
    from objectstate import ObjectState
    from openhcs.core.config import PathPlanningConfig, PipelineConfig
    from pyqt_reactive.forms.parameter_form_manager import FormManagerConfig, ParameterFormManager
    from pyqt_reactive.theming import ColorScheme
    from pyqt_reactive.windows.help_window_manager import source_dataclass_type

    app = QApplication.instance() or QApplication([])
    manager = ParameterFormManager(
        state=ObjectState(PipelineConfig()),
        config=FormManagerConfig(field_id="", color_scheme=ColorScheme()),
    )
    for _ in range(200):
        app.processEvents()

    nested_manager = manager.nested_managers["path_planning_config"]

    assert source_dataclass_type(nested_manager.function_target) is PathPlanningConfig
    assert source_dataclass_type(
        nested_manager.labels["output_dir_suffix"].help_context.help_target
    ) is PathPlanningConfig
