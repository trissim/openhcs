"""Fold typed CellProfiler artifact-binding tuples into ``setting_bindings``.

Run from the repository root.  The codemod preserves each class body's binding
declaration order and removes the superseded image/object/grid tuple names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import libcst as cst


TYPED_BINDING_NAMES = frozenset(
    {
        "image_input_bindings",
        "object_input_bindings",
        "spatial_grid_input_bindings",
        "image_output_bindings",
        "object_output_bindings",
        "spatial_grid_output_bindings",
    }
)
CANONICAL_BINDING_NAME = "setting_bindings"


def _assigned_name(statement: cst.BaseStatement) -> str | None:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    assignment = statement.body[0]
    target: cst.BaseAssignTargetExpression
    if isinstance(assignment, cst.Assign) and len(assignment.targets) == 1:
        target = assignment.targets[0].target
    elif isinstance(assignment, cst.AnnAssign):
        target = assignment.target
    else:
        return None
    return target.value if isinstance(target, cst.Name) else None


def _assignment_value(statement: cst.BaseStatement) -> cst.BaseExpression:
    assignment = statement.body[0]
    if isinstance(assignment, cst.Assign):
        return assignment.value
    if isinstance(assignment, cst.AnnAssign) and assignment.value is not None:
        return assignment.value
    raise TypeError("Binding declaration must have a value.")


def _tuple_elements(value: cst.BaseExpression) -> tuple[cst.Element, ...]:
    if isinstance(value, cst.Tuple):
        return tuple(value.elements)
    return (cst.StarredElement(value=value),)


class BindingDeclarationTransformer(cst.CSTTransformer):
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        body = tuple(updated_node.body.body)
        declarations = tuple(
            (index, _assigned_name(statement), statement)
            for index, statement in enumerate(body)
            if _assigned_name(statement)
            in (*TYPED_BINDING_NAMES, CANONICAL_BINDING_NAME)
        )
        typed = tuple(
            declaration
            for declaration in declarations
            if declaration[1] in TYPED_BINDING_NAMES
        )
        if not typed:
            return updated_node

        insertion_index = min(index for index, _name, _statement in declarations)
        elements = tuple(
            element
            for _index, _name, statement in declarations
            for element in _tuple_elements(_assignment_value(statement))
        )
        canonical = cst.SimpleStatementLine(
            body=(
                cst.Assign(
                    targets=(cst.AssignTarget(cst.Name(CANONICAL_BINDING_NAME)),),
                    value=cst.Tuple(elements=elements),
                ),
            )
        )
        declaration_indexes = frozenset(
            index for index, _name, _statement in declarations
        )
        new_body: list[cst.BaseStatement] = []
        for index, statement in enumerate(body):
            if index == insertion_index:
                new_body.append(canonical)
            if index not in declaration_indexes:
                new_body.append(statement)
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=tuple(new_body))
        )


def migrate(path: Path) -> bool:
    source = path.read_text()
    updated = cst.parse_module(source).visit(BindingDeclarationTransformer()).code
    if updated == source:
        return False
    path.write_text(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    changed = tuple(path for path in args.paths if migrate(path))
    print(f"migrated {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
