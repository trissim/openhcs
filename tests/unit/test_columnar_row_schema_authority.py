"""Static enforcement for annotation-owned dataclass row schemas."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]


def _name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return None


def _semantic_name(node: ast.AST) -> str | None:
    """Resolve exact static schema names without importing production modules."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "value"
        and isinstance(node.value, ast.Attribute)
    ):
        return node.value.attr.lower()
    return None


def _instance_annotations(node: ast.ClassDef) -> frozenset[str]:
    return frozenset(
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and "ClassVar" not in ast.unparse(statement.annotation)
    )


def _assigned_value(
    statement: ast.stmt,
    target_name: str,
) -> ast.AST | None:
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == target_name
    ):
        return statement.value
    if isinstance(statement, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == target_name
        for target in statement.targets
    ):
        return statement.value
    return None


def _field_spec_names(node: ast.ClassDef) -> frozenset[str]:
    values = tuple(
        value
        for statement in node.body
        if (value := _assigned_value(statement, "fields")) is not None
    )
    return frozenset(
        field_name
        for value in values
        for call in ast.walk(value)
        if isinstance(call, ast.Call)
        and _name(call.func) == "FieldSpec"
        and call.args
        and (field_name := _semantic_name(call.args[0])) is not None
    )


def _column_mapping_names(node: ast.ClassDef) -> frozenset[str]:
    column_methods = tuple(
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "columns"
    )
    return frozenset(
        field_name
        for method in column_methods
        for mapping in ast.walk(method)
        if isinstance(mapping, ast.Dict)
        for key in mapping.keys
        if key is not None and (field_name := _semantic_name(key)) is not None
    )


def _direct_column_projection_names(node: ast.ClassDef) -> frozenset[str]:
    """Find row fields projected directly into a class-owned columns mapping."""

    column_methods = tuple(
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        and statement.name == "columns"
    )
    return frozenset(
        attribute.attr
        for method in column_methods
        for mapping in ast.walk(method)
        if isinstance(mapping, ast.Dict)
        for value in mapping.values
        for attribute in ast.walk(value)
        if isinstance(attribute, ast.Attribute)
        and isinstance(attribute.value, ast.Name)
        and attribute.value.id == "self"
    )


def _mirrored_dataclass_fields(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    mirrors: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not any(
            _name(decorator) == "dataclass" for decorator in node.decorator_list
        ):
            continue
        annotations = _instance_annotations(node)
        repeated = annotations.intersection(
            _field_spec_names(node)
            | _column_mapping_names(node)
            | _direct_column_projection_names(node)
        )
        mirrors.extend((node.name, field_name) for field_name in sorted(repeated))
    return tuple(mirrors)


def test_semantic_mirror_detector_rejects_repeated_dataclass_row_schema() -> None:
    tree = ast.parse(
        """
@dataclass
class MirroredRows(ColumnarRows):
    value: int
    fields = (FieldSpec("value", int),)

    @property
    def columns(self):
        return {"value": (self.value,)}
"""
    )

    assert _mirrored_dataclass_fields(tree) == (("MirroredRows", "value"),)


def test_semantic_mirror_detector_resolves_nominal_field_names() -> None:
    tree = ast.parse(
        """
@dataclass
class MirroredRows(ColumnarRows):
    slice_index: int
    result_value: float
    fields = (
        FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
        FieldSpec(MeasurementRowValueField.RESULT_VALUE.value, float),
    )

    @property
    def columns(self):
        return {
            MeasurementRowAxisField.SLICE_INDEX.value: (self.slice_index,),
            MeasurementRowValueField.RESULT_VALUE.value: (self.result_value,),
        }
"""
    )

    assert _mirrored_dataclass_fields(tree) == (
        ("MirroredRows", "result_value"),
        ("MirroredRows", "slice_index"),
    )


def test_production_dataclass_rows_do_not_mirror_schema_or_columns() -> None:
    violations = tuple(
        (path.relative_to(PROJECT_ROOT), class_name, field_name)
        for path in (PROJECT_ROOT / "openhcs").rglob("*.py")
        for class_name, field_name in _mirrored_dataclass_fields(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
    )

    assert violations == ()
