"""Migrate stale RuntimeInputBindingRequest facade consumers."""

from __future__ import annotations

import argparse
from pathlib import Path

import libcst as cst


DEFAULT_PATHS = (
    Path("openhcs/interop/cellprofiler/runtime/object_input_policies.py"),
    Path("openhcs/interop/cellprofiler/runtime/object_measurement_vectors.py"),
    Path("openhcs/processing/backends/cellprofiler/classification.py"),
    Path("openhcs/processing/backends/cellprofiler/colocalization.py"),
    Path("openhcs/processing/backends/cellprofiler/display_modules.py"),
    Path("openhcs/processing/backends/cellprofiler/intensity_distribution.py"),
    Path("openhcs/processing/backends/cellprofiler/measurement_math.py"),
    Path("openhcs/processing/backends/cellprofiler/object_filtering.py"),
)


def _is_request_expression(node: cst.BaseExpression) -> bool:
    return (
        isinstance(node, cst.Name)
        and node.value == "request"
        or (
            isinstance(node, cst.Attribute)
            and isinstance(node.value, cst.Name)
            and node.value.value == "self"
            and node.attr.value == "request"
        )
    )


def _callable_contract_request(
    node: cst.BaseExpression,
) -> cst.BaseExpression | None:
    if (
        isinstance(node, cst.Attribute)
        and node.attr.value == "callable_contract"
        and _is_request_expression(node.value)
    ):
        return node.value
    return None


class RuntimeInputBindingRequestTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.contract_replacements = 0
        self.runtime_input_replacements = 0
        self.collection_replacements = 0

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        del original_node
        if _is_request_expression(updated_node.value):
            if updated_node.attr.value == "contract":
                self.contract_replacements += 1
                return updated_node.with_changes(attr=cst.Name("callable_contract"))
            if updated_node.attr.value == "runtime_inputs":
                self.runtime_input_replacements += 1
                return updated_node.with_changes(attr=cst.Name("declared_inputs"))

        request = _callable_contract_request(updated_node.value)
        if request is None:
            return updated_node
        if updated_node.attr.value == "declared_inputs":
            self.collection_replacements += 1
            return cst.Attribute(value=request, attr=cst.Name("declared_inputs"))
        if updated_node.attr.value == "declared_outputs":
            self.collection_replacements += 1
            return updated_node.with_changes(attr=cst.Name("artifact_outputs"))
        return updated_node


def migrate(path: Path) -> tuple[int, int, int]:
    source = path.read_text(encoding="utf-8")
    transformer = RuntimeInputBindingRequestTransformer()
    updated = cst.parse_module(source).visit(transformer).code
    if updated != source:
        path.write_text(updated, encoding="utf-8")
    return (
        transformer.contract_replacements,
        transformer.runtime_input_replacements,
        transformer.collection_replacements,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    totals = [0, 0, 0]
    for path in args.paths or DEFAULT_PATHS:
        counts = migrate(path)
        totals = [total + count for total, count in zip(totals, counts, strict=True)]
        if any(counts):
            print(
                f"{path}: replaced {counts[0]} contract access(es), "
                f"{counts[1]} runtime-input access(es), and "
                f"{counts[2]} collection access(es)"
            )
    print(
        f"replaced {totals[0]} contract access(es), "
        f"{totals[1]} runtime-input access(es), and "
        f"{totals[2]} collection access(es)"
    )


if __name__ == "__main__":
    main()
