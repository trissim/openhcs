"""Move typed CellProfiler artifact hooks onto the canonical binding-owned API."""

from __future__ import annotations

import argparse
from pathlib import Path

import libcst as cst


METHOD_RENAMES = {
    "artifact_input_names_from_setting": "artifact_names_for_binding",
    "image_output_artifact_relations": "artifact_output_relations",
    "spatial_grid_output_relations": "artifact_output_relations",
}


class ArtifactOutputHookTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.renamed_function_depth = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if node.name.value in METHOD_RENAMES:
            self.renamed_function_depth += 1
        return True

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        replacement = METHOD_RENAMES.get(original_node.name.value)
        if replacement is None:
            return updated_node
        self.renamed_function_depth -= 1
        parameters = tuple(
            parameter.with_changes(name=cst.Name("binding"))
            if parameter.name.value == "setting"
            else parameter
            for parameter in updated_node.params.params
        )
        return updated_node.with_changes(
            name=cst.Name(replacement),
            params=updated_node.params.with_changes(params=parameters),
        )

    def leave_Name(
        self,
        original_node: cst.Name,
        updated_node: cst.Name,
    ) -> cst.Name:
        if self.renamed_function_depth and original_node.value == "setting":
            return updated_node.with_changes(value="binding")
        return updated_node

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.Attribute:
        replacement = METHOD_RENAMES.get(original_node.attr.value)
        if replacement is None:
            return updated_node
        return updated_node.with_changes(attr=cst.Name(replacement))

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.Call:
        if not isinstance(original_node.func, cst.Attribute):
            return updated_node
        if original_node.func.attr.value not in {
            "image_output_artifact_relations",
            "spatial_grid_output_relations",
        }:
            return updated_node
        return updated_node.with_changes(
            args=tuple(
                argument.with_changes(keyword=cst.Name("binding"))
                if argument.keyword is not None
                and argument.keyword.value == "setting"
                else argument
                for argument in updated_node.args
            )
        )


def migrate(path: Path) -> bool:
    source = path.read_text()
    updated = cst.parse_module(source).visit(ArtifactOutputHookTransformer()).code
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
