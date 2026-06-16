"""Parameter help should use CellProfiler function documentation."""


def test_cellprofiler_parameter_help_uses_function_docstring() -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from openhcs.processing.backends import cellprofiler
    from pyqt_reactive.windows.help_window_manager import (
        resolved_parameter_description,
    )

    description = resolved_parameter_description(
        help_target=cellprofiler.measure_object_intensity,
        param_name="labels",
        widget_description="Parameter: Labels",
    )

    assert "ObjectIntensityLabelInput" in description
    assert "CellProfiler MeasureObjectIntensity execution" in description
    assert description != "Parameter: Labels"


def test_cellprofiler_parameter_help_content_hides_raw_union_annotation() -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from openhcs.processing.backends import cellprofiler
    from pyqt_reactive.windows.help_window_manager import (
        parameter_help_content,
        resolved_parameter_description,
    )

    description = resolved_parameter_description(
        help_target=cellprofiler.measure_object_size_shape,
        param_name="shape_backend_provider",
        widget_description="Parameter: Shape Backend Provider",
    )
    content = parameter_help_content(
        param_name="shape_backend_provider",
        param_type=None,
        description=description,
    )

    assert content.summary == "• shape_backend_provider (BackendProviderInput)"
    assert "Default: DefaultCellProfilerBackendProviderSelection()" in content.description
    assert "Controls shape backend provider" in content.description
    assert "openhcs.processing.backends" not in content.summary


def test_cellprofiler_parameter_help_formats_long_setting_docs() -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from openhcs.processing.backends import cellprofiler
    from pyqt_reactive.windows.help_window_manager import (
        parameter_help_content,
        resolved_parameter_description,
    )

    description = resolved_parameter_description(
        help_target=cellprofiler.threshold,
        param_name="threshold_method",
        widget_description="Parameter: Threshold Method",
    )
    content = parameter_help_content(
        param_name="threshold_method",
        param_type=None,
        description=description,
    )

    assert content.summary == "• threshold_method (ThresholdMethod | str)"
    assert "Default: 'Otsu'\n\nCellProfiler setting: Thresholding method" in content.description
    assert "\n\n- {TM_OTSU}:" in content.description
    assert ".. image::" not in content.description


def test_cellprofiler_parameter_help_deduplicates_default_sentences() -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from openhcs.processing.backends import cellprofiler
    from pyqt_reactive.windows.help_window_manager import (
        parameter_help_content,
        resolved_parameter_description,
    )

    description = resolved_parameter_description(
        help_target=cellprofiler.gaussian_filter,
        param_name="sigma",
        widget_description="Parameter: Sigma",
    )
    content = parameter_help_content(
        param_name="sigma",
        param_type=None,
        description=description,
    )

    assert content.summary == "• sigma (float)"
    assert content.description.count("Default: 1.0") == 1
    assert "Default is 1.0" not in content.description
    assert "CellProfiler setting: Sigma" in content.description


def test_parameter_help_long_content_uses_readable_window_width() -> None:
    import openhcs  # noqa: F401 - activates source-checkout externals first
    from openhcs.processing.backends import cellprofiler
    from pyqt_reactive.windows.help_window_manager import (
        HELP_WINDOW_LARGE_WIDTH,
        help_window_width_for_content,
        parameter_help_content,
        resolved_parameter_description,
    )
    from python_introspect.signature_analyzer import DocstringInfo

    description = resolved_parameter_description(
        help_target=cellprofiler.threshold,
        param_name="threshold_method",
        widget_description="Parameter: Threshold Method",
    )
    content = parameter_help_content(
        param_name="threshold_method",
        param_type=None,
        description=description,
    )
    docstring_info = DocstringInfo(
        summary=content.summary,
        description=content.description,
        parameters={},
        returns="",
        examples="",
    )

    assert help_window_width_for_content(docstring_info) == HELP_WINDOW_LARGE_WIDTH


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
