"""Phase 1A boundary tests for declaration-owned CellProfiler policies."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
RUNTIME_ROOT = REPO_ROOT / "openhcs/interop/cellprofiler/runtime"
BACKEND_ROOT = REPO_ROOT / "openhcs/processing/backends/cellprofiler"


def _class_name(node: ast.expr) -> str:
    value = ast.unparse(node).split(".")[-1]
    return value.split("[")[0]


def _class_definitions() -> dict[str, list[tuple[Path, ast.ClassDef]]]:
    definitions: dict[str, list[tuple[Path, ast.ClassDef]]] = {}
    source_paths = {
        *(
            REPO_ROOT / "openhcs/interop/cellprofiler"
        ).rglob("*.py"),
        *BACKEND_ROOT.glob("*.py"),
    }
    for path in sorted(source_paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                definitions.setdefault(node.name, []).append((path, node))
    return definitions


def _module_mro_class_names(
    definitions: dict[str, list[tuple[Path, ast.ClassDef]]],
) -> set[str]:
    module_descendants = {"CellProfilerModule"}
    while True:
        discovered = {
            class_name
            for class_name, class_definitions in definitions.items()
            if any(
                _class_name(base) in module_descendants
                for _path, node in class_definitions
                for base in node.bases
            )
        }
        if discovered <= module_descendants:
            break
        module_descendants.update(discovered)

    ancestors = set(module_descendants)
    while True:
        discovered = {
            _class_name(base)
            for class_name in tuple(ancestors)
            for _path, node in definitions.get(class_name, ())
            for base in node.bases
            if _class_name(base) in definitions
        }
        if discovered <= ancestors:
            return ancestors
        ancestors.update(discovered)


def test_module_policy_authority_roots_own_behavior() -> None:
    definitions = _class_definitions()
    for path, node in (
        definition
        for class_definitions in definitions.values()
        for definition in class_definitions
        if definition[1].name.endswith("PolicyMixin")
        and "CellProfilerModuleAuthority"
        in {_class_name(base) for base in definition[1].bases}
    ):
        methods = tuple(item for item in node.body if isinstance(item, ast.FunctionDef))
        assert methods, (
            f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name} is an "
            "empty policy tag; concrete behavior belongs on its MRO owner"
        )
        for method in methods:
            assert "classmethod" in {
                ast.unparse(decorator) for decorator in method.decorator_list
            }


def test_module_mro_overrides_of_module_classmethods_remain_class_bound() -> None:
    definitions = _class_definitions()
    module_root = definitions["CellProfilerModule"][0][1]
    module_classmethods = {
        method.name
        for method in module_root.body
        if isinstance(method, ast.FunctionDef)
        and "classmethod"
        in {ast.unparse(decorator) for decorator in method.decorator_list}
    }
    for class_name in _module_mro_class_names(definitions):
        for path, node in definitions[class_name]:
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if item.name not in module_classmethods:
                    continue
                decorators = {ast.unparse(value) for value in item.decorator_list}
                assert "classmethod" in decorators, (
                    f"{path.relative_to(REPO_ROOT)}:{item.lineno} "
                    f"{class_name}.{item.name} is not a classmethod"
                )


def test_object_measurement_columnar_rows_own_complete_domain_semantics() -> None:
    from openhcs.processing.backends.cellprofiler.colocalization import (
        ObjectColocalizationColumnarMeasurements,
    )
    from openhcs.processing.backends.cellprofiler.granularity import (
        ObjectGranularityMeasurementRows,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        ObjectIntensityMeasurementRows,
    )
    from openhcs.processing.backends.cellprofiler.object_measurement_columnar_rows import (
        ObjectMeasurementColumnarRows,
    )
    from openhcs.processing.backends.cellprofiler.shape import (
        ShapeObjectMeasurementRows,
    )

    for row_type in (
        ObjectColocalizationColumnarMeasurements,
        ObjectGranularityMeasurementRows,
        ObjectIntensityMeasurementRows,
        ShapeObjectMeasurementRows,
    ):
        assert issubclass(row_type, ObjectMeasurementColumnarRows)
        assert "covers_declared_object_measurement_domain" not in row_type.__dict__


def test_registered_runtime_bodies_do_not_parse_cellprofiler_enum_text() -> None:
    """Legacy CellProfiler text is normalized before registered runtime calls."""

    occurrences: list[str] = []
    for path in BACKEND_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.decorator_list:
                continue
            for descendant in ast.walk(node):
                if (
                    isinstance(descendant, ast.Call)
                    and isinstance(descendant.func, (ast.Name, ast.Attribute))
                    and (
                        descendant.func.id
                        if isinstance(descendant.func, ast.Name)
                        else descendant.func.attr
                    )
                    == "coerce_cellprofiler_enum"
                ):
                    occurrences.append(
                        f"{path.relative_to(REPO_ROOT)}:{descendant.lineno}"
                    )

    assert occurrences == []


def test_enum_text_coercion_remains_at_cellprofiler_source_boundaries() -> None:
    """Runtime computation receives enums; only source parsers accept CP text."""

    module_class_names = _module_mro_class_names(_class_definitions())
    runtime_occurrences: list[str] = []
    parser_name_markers = (
        "cellprofiler",
        "literal",
        "parse",
        "setting",
        "from_module",
    )

    for path in BACKEND_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, (ast.Name, ast.Attribute))
                and (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                )
                == "coerce_cellprofiler_enum"
            ):
                continue

            enclosing_function: ast.FunctionDef | None = None
            enclosing_class: ast.ClassDef | None = None
            ancestor = parents.get(node)
            while ancestor is not None:
                if (
                    enclosing_function is None
                    and isinstance(ancestor, ast.FunctionDef)
                ):
                    enclosing_function = ancestor
                if isinstance(ancestor, ast.ClassDef):
                    enclosing_class = ancestor
                    break
                ancestor = parents.get(ancestor)

            if (
                enclosing_class is not None
                and enclosing_class.name in module_class_names
            ):
                continue
            function_name = (
                enclosing_function.name if enclosing_function is not None else ""
            )
            if any(marker in function_name for marker in parser_name_markers):
                continue
            runtime_occurrences.append(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{function_name}"
            )

    assert runtime_occurrences == []
