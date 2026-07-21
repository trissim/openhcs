"""Move settings translation off the CellProfiler registry root."""

from __future__ import annotations

from pathlib import Path

import libcst as cst


ROOT_PATH = Path("openhcs/interop/cellprofiler/module_declarations.py")
OWNER_PATH = Path("openhcs/interop/cellprofiler/module_settings.py")

TOP_LEVEL_NAMES = frozenset(
    {
        "_enum_type_from_annotation",
        "_coerce_callable_enum_kwarg",
    }
)
ATTRIBUTE_NAMES = frozenset({"setting_bindings", "ignored_settings"})
METHOD_NAMES = frozenset(
    {
        "declared_setting_bindings",
        "normalize_setting_name",
        "_bind_declared_settings",
        "_finalize_bound_settings",
        "_coerce_kwargs_to_callable_signature",
        "bind_settings",
        "postprocess_bound_settings",
        "ignored_settings_for",
        "setting_value",
    }
)


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


def _declaration_name(statement: cst.BaseStatement) -> str | None:
    if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
        return statement.name.value
    return _assigned_name(statement)


def main() -> None:
    root_module = cst.parse_module(ROOT_PATH.read_text())
    moved_top_level: list[cst.BaseStatement] = []
    moved_members: list[cst.BaseStatement] = []
    root_body: list[cst.BaseStatement] = []
    for statement in root_module.body:
        name = _declaration_name(statement)
        if name in TOP_LEVEL_NAMES:
            moved_top_level.append(statement)
            continue
        if not isinstance(statement, cst.ClassDef) or statement.name.value != "CellProfilerModule":
            root_body.append(statement)
            continue
        retained: list[cst.BaseStatement] = []
        for member in statement.body.body:
            member_name = _declaration_name(member)
            if member_name in ATTRIBUTE_NAMES or member_name in METHOD_NAMES:
                moved_members.append(member)
            else:
                retained.append(member)
        bases = (
            cst.Arg(cst.Name("CellProfilerModuleSettings")),
            *statement.bases,
        )
        root_body.append(
            statement.with_changes(
                bases=bases,
                body=statement.body.with_changes(body=tuple(retained)),
            )
        )

    if len(moved_top_level) != len(TOP_LEVEL_NAMES):
        raise RuntimeError(
            f"Expected {len(TOP_LEVEL_NAMES)} helpers, found {len(moved_top_level)}."
        )
    expected_members = len(ATTRIBUTE_NAMES) + len(METHOD_NAMES)
    if len(moved_members) != expected_members:
        raise RuntimeError(
            f"Expected {expected_members} declarations, found {len(moved_members)}."
        )

    owner_module = cst.parse_module(OWNER_PATH.read_text())
    owner_class = cst.ClassDef(
        name=cst.Name("CellProfilerModuleSettings"),
        body=cst.IndentedBlock(body=tuple(moved_members)),
        leading_lines=(cst.EmptyLine(), cst.EmptyLine()),
    )
    ROOT_PATH.write_text(root_module.with_changes(body=tuple(root_body)).code)
    OWNER_PATH.write_text(
        owner_module.with_changes(
            body=(*owner_module.body, *moved_top_level, owner_class)
        ).code
    )


if __name__ == "__main__":
    main()
