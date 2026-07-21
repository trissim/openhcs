"""Replace deleted typed binding attributes with the canonical declaration query."""

from __future__ import annotations

import argparse
from pathlib import Path

import libcst as cst


QUERY_BY_ATTRIBUTE = {
    "image_input_bindings": ("ArtifactInputPlan", "ImageArtifactType"),
    "object_input_bindings": ("ArtifactInputPlan", "ObjectLabelsArtifactType"),
    "spatial_grid_input_bindings": ("ArtifactInputPlan", "SpatialGridArtifactType"),
    "image_output_bindings": ("ArtifactOutputPlan", "ImageArtifactType"),
    "object_output_bindings": ("ArtifactOutputPlan", "ObjectLabelsArtifactType"),
    "spatial_grid_output_bindings": ("ArtifactOutputPlan", "SpatialGridArtifactType"),
}


class TypedBindingQueryTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.required_imports: set[str] = set()

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        if not isinstance(updated_node.value, cst.Name):
            return updated_node
        query = QUERY_BY_ATTRIBUTE.get(updated_node.attr.value)
        if query is None:
            return updated_node
        plan_type, artifact_type = query
        self.required_imports.update((plan_type, artifact_type))
        return cst.Call(
            func=cst.Attribute(
                value=updated_node.value,
                attr=cst.Name("declared_artifact_bindings"),
            ),
            args=(
                cst.Arg(
                    keyword=cst.Name("plan_type"),
                    value=cst.Name(plan_type),
                ),
                cst.Arg(
                    keyword=cst.Name("artifact_type"),
                    value=cst.Name(artifact_type),
                ),
            ),
        )


def _imported_artifact_names(module: cst.Module) -> set[str]:
    imported: set[str] = set()
    for statement in module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if not isinstance(small, cst.ImportFrom):
                continue
            if cst.Module([]).code_for_node(small.module) != "openhcs.core.artifacts":
                continue
            if isinstance(small.names, cst.ImportStar):
                return set(QUERY_BY_ATTRIBUTE.values())
            imported.update(
                alias.name.value
                for alias in small.names
                if isinstance(alias.name, cst.Name)
            )
    return imported


def migrate(path: Path) -> bool:
    source = path.read_text()
    module = cst.parse_module(source)
    transformer = TypedBindingQueryTransformer()
    updated = module.visit(transformer)
    missing = transformer.required_imports - _imported_artifact_names(updated)
    if missing:
        import_statement = cst.parse_statement(
            "from openhcs.core.artifacts import " + ", ".join(sorted(missing)) + "\n"
        )
        body = list(updated.body)
        insertion = 1 if body and isinstance(body[0], cst.SimpleStatementLine) else 0
        while insertion < len(body):
            statement = body[insertion]
            code = updated.code_for_node(statement)
            if code.startswith("from __future__ import"):
                insertion += 1
                continue
            break
        body.insert(insertion, import_statement)
        updated = updated.with_changes(body=tuple(body))
    code = updated.code
    if code == source:
        return False
    path.write_text(code)
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
