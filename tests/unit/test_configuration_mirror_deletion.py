"""Static gates for deleted configuration semantic mirrors."""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
SCANNED_ROOTS = (
    PROJECT_ROOT / "openhcs",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "benchmark",
    PROJECT_ROOT / "external/ObjectState/src",
    PROJECT_ROOT / "external/pyqt-reactive/src",
    PROJECT_ROOT / "external/python-introspect/src",
    PROJECT_ROOT / "external/zmqruntime/src",
    PROJECT_ROOT / "external/PolyStore/src",
)
LEGACY_CONFIG_PREFIX = "openhcs." + "config_framework"
REQUIRED_DELETED_PATHS = (
    "openhcs/config_framework/__init__.py",
    "openhcs/introspection/lazy_dataclass_utils.py",
    "external/pyqt-reactive/src/pyqt_reactive/widgets/shared/"
    "manager_config_resolution.py",
    "external/pyqt-reactive/src/pyqt_reactive/services/zmq_server_info_parser.py",
    "external/pyqt-reactive/src/pyqt_reactive/protocols/zmq_server_protocol.py",
    "openhcs/pyqt_gui/widgets/shared/server_browser/server_tree_population.py",
    "openhcs/core/lazy_placeholder.py",
    "openhcs/utils/display_config_factory.py",
    "openhcs/utils/enum_factory.py",
)


def _python_files():
    for root in SCANNED_ROOTS:
        yield from root.rglob("*.py")


def _embedded_imports(source: str) -> tuple[ast.ImportFrom, ...]:
    """Parse complete import statements embedded in prose/code templates."""

    import_statements: list[ast.ImportFrom] = []
    pattern = re.compile(
        r"(?m)^[ \t]*from\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
        r"\s+import\s+(?:\([^)]*\)|[^\n]+)"
    )
    for match in pattern.finditer(source):
        try:
            parsed = ast.parse(textwrap.dedent(match.group(0)))
        except SyntaxError:
            continue
        if len(parsed.body) == 1 and isinstance(parsed.body[0], ast.ImportFrom):
            import_statements.append(parsed.body[0])
    return tuple(import_statements)


def test_configuration_mirror_modules_remain_deleted() -> None:
    for relative_path in REQUIRED_DELETED_PATHS:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path


def test_false_viewer_options_remain_absent_from_config_declarations() -> None:
    config_path = PROJECT_ROOT / "openhcs/core/config.py"
    tree = ast.parse(config_path.read_text(), filename=str(config_path))
    forbidden_fields = {
        "batch_size",
        "fiji_executable_path",
    }
    occurrences = [
        f"{config_path.relative_to(PROJECT_ROOT)}:{node.lineno}:{node.target.id}"
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in forbidden_fields
    ]

    assert occurrences == []


def test_dtype_conversion_remains_owned_only_by_arraybridge() -> None:
    constants_path = PROJECT_ROOT / "openhcs/constants/constants.py"
    constants_tree = ast.parse(
        constants_path.read_text(),
        filename=str(constants_path),
    )
    forbidden_declarations = {
        node.name
        for node in constants_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"DtypeConversion", "LiteralDtype"}
    }
    forbidden_assignments = {
        target.id
        for node in constants_tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else (node.target,)
        )
        if isinstance(target, ast.Name)
        and target.id in {"DtypeConversion", "LiteralDtype"}
    }
    forbidden_imports: list[str] = []
    forbidden_embedded_imports: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                in {
                    "openhcs.constants",
                    "openhcs.constants.constants",
                    "openhcs.core.memory",
                }
                and any(
                    alias.name in {"DtypeConversion", "LiteralDtype"}
                    for alias in node.names
                )
            ):
                forbidden_imports.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                )
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
            ):
                for embedded_import in _embedded_imports(node.value):
                    if (
                        embedded_import.module
                        in {
                            "openhcs.constants",
                            "openhcs.constants.constants",
                            "openhcs.core.memory",
                        }
                        and any(
                            alias.name in {"DtypeConversion", "LiteralDtype"}
                            for alias in embedded_import.names
                        )
                    ):
                        forbidden_embedded_imports.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                        )

    assert forbidden_declarations == set()
    assert forbidden_assignments == set()
    assert forbidden_imports == []
    assert forbidden_embedded_imports == []


def test_cpu_cell_counter_has_no_false_segmentation_output_toggle() -> None:
    path = (
        PROJECT_ROOT
        / "openhcs/processing/backends/analysis/cell_counting_cpu.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "count_cells_single_channel"
    )
    parameter_names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }

    assert "return_segmentation_mask" not in parameter_names


def test_experimental_analysis_is_not_a_global_pipeline_config() -> None:
    config_path = PROJECT_ROOT / "openhcs/core/config.py"
    tree = ast.parse(config_path.read_text(), filename=str(config_path))
    analysis_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ExperimentalAnalysisConfig"
    )
    decorators = {ast.unparse(decorator) for decorator in analysis_config.decorator_list}

    assert not any(
        decorator == "global_pipeline_config"
        or decorator.startswith("global_pipeline_config(")
        for decorator in decorators
    )


def test_legacy_config_namespace_imports_do_not_recur() -> None:
    occurrences: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                LEGACY_CONFIG_PREFIX
            ):
                occurrences.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                occurrences.extend(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                    for alias in node.names
                    if alias.name.startswith(LEGACY_CONFIG_PREFIX)
                )
    assert occurrences == []


def test_deleted_lazy_registry_and_name_resolver_symbols_do_not_recur() -> None:
    forbidden_names = {
        "register_" + "lazy_type",
        "get_registered_" + "lazy_types",
        "_openhcs_" + "type_resolver",
        "discover_" + "lazy_dataclass_types",
        "ManagerGui" + "ConfigResolution",
        "_LAZY_" + "TYPE_REGISTRY",
        "_migrate_" + "dataclass",
        "_nested_" + "dataclass_type",
        "_has_" + "lazy_resolution",
        "_inherit_" + "as_none",
        "_is_" + "lazy_dataclass",
        "_unwrap_" + "optional_type",
        "_load_or_" + "discover_functions",
        "Parameter" + "Analyzer",
    }
    occurrences: list[str] = []
    scanned_roots = (
        PROJECT_ROOT / "openhcs",
        PROJECT_ROOT / "external/ObjectState/src",
        PROJECT_ROOT / "external/pyqt-reactive/src",
    )
    for root in scanned_roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Name, ast.Attribute)):
                    name = node.id if isinstance(node, ast.Name) else node.attr
                    if name in forbidden_names:
                        occurrences.append(
                            f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                        )
    assert occurrences == []


def test_python_introspection_compatibility_aliases_remain_deleted() -> None:
    path = (
        PROJECT_ROOT
        / "external/python-introspect/src/python_introspect/"
        "unified_parameter_analyzer.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden = {"Parameter" + "Analyzer", "analyze_" + "parameters"}
    declarations = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assignments = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else (node.target,)
        )
        if isinstance(target, ast.Name)
    }
    assert not forbidden & (declarations | assignments)


def test_zmq_transport_semantics_have_one_nominal_owner() -> None:
    transport_definitions: list[str] = []
    forbidden_symbols = {
        "coerce_" + "transport_mode",
        "Viewer" + "TransportMode",
        "OpenHCS" + "TransportMode",
    }
    forbidden_occurrences: list[str] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TransportMode":
                transport_definitions.append(str(path.relative_to(PROJECT_ROOT)))
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name in forbidden_symbols:
                    forbidden_occurrences.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                    )

    assert transport_definitions == ["external/zmqruntime/src/zmqruntime/config.py"]
    assert forbidden_occurrences == []


def test_ui_bridge_sparse_projection_has_no_field_name_protocol() -> None:
    scanned_paths = (
        PROJECT_ROOT / "openhcs/agent/dto/ui_bridge.py",
        PROJECT_ROOT / "openhcs/mcp/server.py",
    )
    forbidden_name = "connection_" + "fields"
    occurrences: list[str] = []

    for path in scanned_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name == forbidden_name:
                    occurrences.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                    )

    assert occurrences == []


def test_typed_transport_boundaries_do_not_regrow_payload_schema_helpers() -> None:
    forbidden_names = {
        "RuntimeServer" + "Payload",
        "UiBridgeDescriptor" + "Payload",
        "JsonDescriptorValue" + "Authority",
        "EnvironmentValue" + "Authority",
        "zmq_client_" + "kwargs",
    }
    occurrences: list[str] = []

    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute, ast.ClassDef)):
                if isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                else:
                    name = node.name
                if name in forbidden_names:
                    occurrences.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                    )

    assert occurrences == []


def test_system_monitor_keeps_metrics_and_fixed_colors_typed() -> None:
    scanned_paths = (
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/services/"
        "system_metrics_sampler.py",
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/services/"
        "system_monitor_config.py",
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/services/"
        "system_monitor_core.py",
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/services/"
        "persistent_system_monitor.py",
        PROJECT_ROOT
        / "external/pyqt-reactive/src/pyqt_reactive/widgets/system_monitor.py",
    )
    forbidden_names = {
        "chart_" + "colors",
        "get_metrics_" + "dict",
    }
    occurrences: list[str] = []

    for path in scanned_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute, ast.FunctionDef)):
                if isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                else:
                    name = node.name
                if name in forbidden_names:
                    occurrences.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                    )

    assert occurrences == []


def test_zmq_server_browser_keeps_pong_and_tree_identity_typed() -> None:
    forbidden_names = {
        "DefaultServerInfo" + "Parser",
        "ServerInfoParser" + "ABC",
        "RunningExecution" + "Entry",
        "QueuedExecution" + "Entry",
        "DictPayloadTreeItem" + "KeyBuilder",
        "ServerTree" + "Population",
    }
    scanned_paths = (
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/services/"
        "zmq_server_info.py",
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/services/"
        "zmq_server_scan_service.py",
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/widgets/shared/"
        "zmq_server_browser_widget.py",
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/widgets/shared/"
        "tree_sync_adapter.py",
        PROJECT_ROOT / "external/pyqt-reactive/src/pyqt_reactive/widgets/shared/"
        "tree_state_adapter.py",
        PROJECT_ROOT / "openhcs/pyqt_gui/widgets/shared/zmq_server_manager.py",
        PROJECT_ROOT
        / "openhcs/pyqt_gui/widgets/shared/server_browser/live_tree_sync.py",
    )
    occurrences: list[str] = []

    for path in scanned_paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute, ast.ClassDef)):
                if isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                else:
                    name = node.name
                if name in forbidden_names or name == "raw":
                    occurrences.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{name}"
                    )

    assert occurrences == []


def test_viewer_identity_is_owned_by_the_streaming_declaration_type() -> None:
    definitions: list[str] = []
    stale_imports: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ViewerType":
                definitions.append(str(path.relative_to(PROJECT_ROOT)))
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "openhcs.runtime.viewer_protocol"
                and any(alias.name == "ViewerType" for alias in node.names)
            ):
                stale_imports.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                )

    assert definitions == [
        "openhcs/core/streaming_config_declarations.py"
    ]
    assert stale_imports == []

    declarations_path = (
        PROJECT_ROOT / "openhcs/core/streaming_config_declarations.py"
    )
    declarations_tree = ast.parse(
        declarations_path.read_text(),
        filename=str(declarations_path),
    )
    assert not any(
        isinstance(node, ast.ClassDef) and node.name == "StreamingViewerConfigSpec"
        for node in declarations_tree.body
    )
    viewer_type = next(
        node
        for node in declarations_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ViewerType"
    )
    members = {
        node.targets[0].id: node.value
        for node in viewer_type.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"FIJI", "NAPARI"}
    }
    assert set(members) == {"FIJI", "NAPARI"}
    assert all(
        isinstance(value, ast.Tuple)
        and len(value.elts) == 2
        and isinstance(value.elts[1], ast.Name)
        and value.elts[1].id.endswith("ViewerDeclaration")
        for value in members.values()
    )
