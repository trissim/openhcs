"""Place ``setting_bindings`` after every class-local binding it references."""

from __future__ import annotations

import argparse
from pathlib import Path

import libcst as cst


def _assigned_name(statement: cst.BaseStatement) -> str | None:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return None
    assignment = statement.body[0]
    if isinstance(assignment, cst.Assign) and len(assignment.targets) == 1:
        target = assignment.targets[0].target
    elif isinstance(assignment, cst.AnnAssign):
        target = assignment.target
    else:
        return None
    return target.value if isinstance(target, cst.Name) else None


def _assignment_value(statement: cst.BaseStatement) -> cst.BaseExpression | None:
    assignment = statement.body[0]
    if isinstance(assignment, cst.Assign):
        return assignment.value
    if isinstance(assignment, cst.AnnAssign):
        return assignment.value
    return None


class ReferencedNames(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: cst.Name) -> None:
        self.names.add(node.value)


class SettingBindingOrderTransformer(cst.CSTTransformer):
    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        body = list(updated_node.body.body)
        binding_index = next(
            (
                index
                for index, statement in enumerate(body)
                if _assigned_name(statement) == "setting_bindings"
            ),
            None,
        )
        if binding_index is None:
            return updated_node
        value = _assignment_value(body[binding_index])
        if value is None:
            return updated_node
        referenced = ReferencedNames()
        value.visit(referenced)
        dependency_indexes = tuple(
            index
            for index, statement in enumerate(body)
            if _assigned_name(statement) in referenced.names
            and _assigned_name(statement) != "setting_bindings"
        )
        if not dependency_indexes or binding_index > max(dependency_indexes):
            return updated_node
        binding_statement = body.pop(binding_index)
        last_dependency = max(
            index
            for index, statement in enumerate(body)
            if _assigned_name(statement) in referenced.names
        )
        body.insert(last_dependency + 1, binding_statement)
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=tuple(body))
        )


def migrate(path: Path) -> bool:
    source = path.read_text()
    updated = cst.parse_module(source).visit(SettingBindingOrderTransformer()).code
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
