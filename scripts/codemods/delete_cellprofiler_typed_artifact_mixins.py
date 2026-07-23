"""Delete superseded typed CellProfiler artifact mixins and their uses."""

from __future__ import annotations

import argparse
from pathlib import Path

import libcst as cst
from libcst import RemovalSentinel


REMOVED = frozenset(
    {
        "ArtifactInputBindingModule",
        "ImageArtifactInputModule",
        "ImageArtifactOutputModule",
        "SpatialGridArtifactInputModule",
        "SpatialGridArtifactOutputModule",
    }
)


class TypedArtifactMixinDeletion(cst.CSTTransformer):
    def __init__(self, *, delete_definitions: bool) -> None:
        self.delete_definitions = delete_definitions

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef | RemovalSentinel:
        if self.delete_definitions and original_node.name.value in REMOVED:
            return cst.RemoveFromParent()
        bases = tuple(
            base
            for base in updated_node.bases
            if not (
                isinstance(base.value, cst.Name)
                and base.value.value in REMOVED
            )
        )
        if updated_node.bases and not bases:
            bases = (cst.Arg(cst.Name("CellProfilerModule")),)
        return updated_node.with_changes(bases=bases)

    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.ImportFrom | RemovalSentinel:
        if isinstance(updated_node.names, cst.ImportStar):
            return updated_node
        names = tuple(
            alias
            for alias in updated_node.names
            if not (
                isinstance(alias.name, cst.Name)
                and alias.name.value in REMOVED
            )
        )
        if not names:
            return cst.RemoveFromParent()
        return updated_node.with_changes(names=names)


def migrate(path: Path, *, delete_definitions: bool) -> bool:
    source = path.read_text()
    updated = cst.parse_module(source).visit(
        TypedArtifactMixinDeletion(delete_definitions=delete_definitions)
    ).code
    if updated == source:
        return False
    path.write_text(updated)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("owner", type=Path)
    parser.add_argument("consumers", nargs="*", type=Path)
    args = parser.parse_args()
    changed = []
    if migrate(args.owner, delete_definitions=True):
        changed.append(args.owner)
    for path in args.consumers:
        if migrate(path, delete_definitions=False):
            changed.append(path)
    print(f"migrated {len(changed)} files")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
