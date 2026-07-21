"""Delete the forwarding-only CellProfiler special-input policy base."""

from __future__ import annotations

import argparse
from pathlib import Path

import libcst as cst


POLICY_MODULE = "openhcs.interop.cellprofiler.runtime.special_input_policies"
POLICY_NAME = "CellProfilerSpecialInputPolicyMixin"


def _dotted_name(node: cst.BaseExpression) -> str | None:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        owner = _dotted_name(node.value)
        return f"{owner}.{node.attr.value}" if owner else None
    return None


class SpecialInputPolicyTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.removed_imports = 0
        self.removed_bases = 0

    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.ImportFrom | cst.RemovalSentinel:
        if (
            original_node.module is None
            or _dotted_name(original_node.module) != POLICY_MODULE
            or isinstance(updated_node.names, cst.ImportStar)
        ):
            return updated_node
        names = tuple(
            alias
            for alias in updated_node.names
            if not (
                isinstance(alias.name, cst.Name) and alias.name.value == POLICY_NAME
            )
        )
        removed = len(updated_node.names) - len(names)
        self.removed_imports += removed
        if not names:
            return cst.RemoveFromParent()
        return updated_node.with_changes(names=names)

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        bases = tuple(
            base
            for base in updated_node.bases
            if not (
                isinstance(base.value, cst.Name) and base.value.value == POLICY_NAME
            )
        )
        self.removed_bases += len(updated_node.bases) - len(bases)
        if not bases and not updated_node.keywords:
            return updated_node.with_changes(
                bases=bases,
                lpar=cst.MaybeSentinel.DEFAULT,
                rpar=cst.MaybeSentinel.DEFAULT,
            )
        return updated_node.with_changes(bases=bases)


def migrate(path: Path) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    transformer = SpecialInputPolicyTransformer()
    updated = cst.parse_module(source).visit(transformer).code
    if updated != source:
        path.write_text(updated, encoding="utf-8")
    return transformer.removed_imports, transformer.removed_bases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    total_imports = 0
    total_bases = 0
    for path in args.paths:
        removed_imports, removed_bases = migrate(path)
        total_imports += removed_imports
        total_bases += removed_bases
        if removed_imports or removed_bases:
            print(
                f"{path}: removed {removed_imports} import(s), {removed_bases} base(s)"
            )
    print(f"removed {total_imports} import(s), {total_bases} base(s)")


if __name__ == "__main__":
    main()
