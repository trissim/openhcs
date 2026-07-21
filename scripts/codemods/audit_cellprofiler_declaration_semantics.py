"""Audit CellProfiler production declarations for collapsed semantic mirrors."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_ROOTS = (
    PROJECT_ROOT / "openhcs/processing/backends/cellprofiler",
    PROJECT_ROOT / "openhcs/interop/cellprofiler",
)
REFLECTION_CALLS = frozenset({"getattr", "setattr", "hasattr", "vars"})
DELETED_API_NAMES = frozenset(
    {
        "ArtifactInputContractPartition",
        "ArtifactInputPartitionStrategy",
        "ArtifactOutputContractPartition",
        "CellProfilerSpecialInputPolicyMixin",
        "DeclaredArtifactOutputPartition",
        "ModuleArtifactContract",
        "_main_flow_artifact_input_names",
        "special_image_inputs",
        "special_input_specs",
    }
)
SEMANTIC_NAME_PARTS = (
    "artifact",
    "function_name",
    "image",
    "measurement",
    "module",
    "object",
    "relation",
    "spec",
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    scope: str
    name: str
    source: str
    detail: str = ""


@dataclass(frozen=True)
class MethodRecord:
    path: Path
    class_name: str
    bases: tuple[str, ...]
    node: ast.FunctionDef | ast.AsyncFunctionDef
    source_lines: tuple[str, ...]

    @property
    def location(self) -> Finding:
        return Finding(
            path=str(self.path.relative_to(PROJECT_ROOT)),
            line=self.node.lineno,
            column=self.node.col_offset,
            scope=self.class_name,
            name=self.node.name,
            source=self.source_lines[self.node.lineno - 1].strip(),
        )


def _dotted_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return None


def _assigned_names(node: ast.AST) -> Iterator[ast.Name]:
    if isinstance(node, ast.Name):
        yield node
    elif isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            yield from _assigned_names(element)


def _without_docstring(
    body: list[ast.stmt],
) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_trivial_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = _without_docstring(node.body)
    if not body:
        return True
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return statement.value.value is Ellipsis
    if isinstance(statement, ast.Raise):
        return (
            isinstance(statement.exc, ast.Call)
            and _dotted_name(statement.exc.func) == "NotImplementedError"
        )
    return False


def _same_name_argument(node: ast.AST, name: str, *, starred: bool = False) -> bool:
    if starred:
        return (
            isinstance(node, ast.Starred)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
        )
    return isinstance(node, ast.Name) and node.id == name


def _unchanged_super_call(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    body = _without_docstring(node.body)
    if len(body) != 1:
        return None
    statement = body[0]
    value: ast.AST | None
    if isinstance(statement, ast.Return):
        value = statement.value
    elif isinstance(statement, ast.Expr):
        value = statement.value
    else:
        return None
    if isinstance(value, ast.Await):
        value = value.value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
        return None
    super_call = value.func.value
    if (
        value.func.attr != node.name
        or not isinstance(super_call, ast.Call)
        or _dotted_name(super_call.func) != "super"
    ):
        return None

    positional = [*node.args.posonlyargs, *node.args.args]
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    expected_positional = [argument.arg for argument in positional]
    actual_positional = list(value.args)
    if node.args.vararg is not None:
        expected_positional.append(node.args.vararg.arg)
    if len(actual_positional) != len(expected_positional):
        return None
    for index, (argument, name) in enumerate(
        zip(actual_positional, expected_positional, strict=True)
    ):
        if not _same_name_argument(
            argument,
            name,
            starred=node.args.vararg is not None
            and index == len(expected_positional) - 1,
        ):
            return None

    expected_keywords = {argument.arg for argument in node.args.kwonlyargs}
    if node.args.kwarg is not None:
        expected_keywords.add(node.args.kwarg.arg)
    actual_keywords: set[str] = set()
    for keyword in value.keywords:
        if keyword.arg is None:
            if node.args.kwarg is None or not _same_name_argument(
                keyword.value, node.args.kwarg.arg
            ):
                return None
            actual_keywords.add(node.args.kwarg.arg)
        elif not _same_name_argument(keyword.value, keyword.arg):
            return None
        else:
            actual_keywords.add(keyword.arg)
    return value if actual_keywords == expected_keywords else None


def _normalized_method(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    normalized = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=_without_docstring(node.body),
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=getattr(node, "type_params", []),
    )
    return ast.dump(normalized, include_attributes=False)


class FileAudit(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source_lines = tuple(source.splitlines())
        self.scope: list[str] = []
        self.findings: dict[str, list[Finding]] = defaultdict(list)
        self.methods: list[MethodRecord] = []

    def _record(
        self,
        category: str,
        node: ast.AST,
        name: str,
        detail: str = "",
    ) -> None:
        self.findings[category].append(
            Finding(
                path=str(self.path.relative_to(PROJECT_ROOT)),
                line=node.lineno,
                column=node.col_offset,
                scope=".".join(self.scope) or "<module>",
                name=name,
                source=self.source_lines[node.lineno - 1].strip(),
                detail=detail,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        bases = tuple(filter(None, (_dotted_name(base) for base in node.bases)))
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.methods.append(
                    MethodRecord(
                        path=self.path,
                        class_name=".".join(self.scope),
                        bases=bases,
                        node=statement,
                        source_lines=self.source_lines,
                    )
                )
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.scope and _unchanged_super_call(node) is not None:
            self._record("forwarding_override", node, node.name)
        if node.name.endswith(("_refs", "_specs")):
            self._record("mirror_declaration", node, node.name, "function")
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name in _assigned_names(target):
                if name.id.endswith(("_refs", "_specs")):
                    self._record("mirror_declaration", name, name.id, "assignment")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name in _assigned_names(node.target):
            if name.id.endswith(("_refs", "_specs")):
                self._record("mirror_declaration", name, name.id, "annotation")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in DELETED_API_NAMES or (
            "Partition" in node.id and "Artifact" in node.id
        ):
            self._record("deleted_api", node, node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in DELETED_API_NAMES or (
            "Partition" in node.attr and "Artifact" in node.attr
        ):
            self._record("deleted_api", node, node.attr)
        if node.attr in {"contract", "runtime_inputs"} and _dotted_name(node.value) in {
            "request",
            "self.request",
        }:
            self._record(
                "stale_runtime_input_request_api",
                node,
                node.attr,
                _dotted_name(node) or "",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        called_name = _dotted_name(node.func)
        simple_name = called_name.rsplit(".", maxsplit=1)[-1] if called_name else ""
        if simple_name in REFLECTION_CALLS:
            self._record("semantic_reflection", node, simple_name, called_name or "")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        values = (node.left, *node.comparators)
        if any(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in values
        ):
            expression_names = " ".join(
                child.id if isinstance(child, ast.Name) else child.attr
                for value in values
                for child in ast.walk(value)
                if isinstance(child, (ast.Name, ast.Attribute))
            ).lower()
            if any(part in expression_names for part in SEMANTIC_NAME_PARTS):
                self._record(
                    "semantic_string_compare",
                    node,
                    ast.unparse(node),
                    expression_names,
                )
        self.generic_visit(node)


def _source_paths(roots: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(
        sorted(path for root in roots for path in root.rglob("*.py") if path.is_file())
    )


def audit(paths: Iterable[Path]) -> dict[str, object]:
    findings: dict[str, list[Finding]] = defaultdict(list)
    methods: list[MethodRecord] = []
    scanned_paths = tuple(paths)
    for path in scanned_paths:
        source = path.read_text(encoding="utf-8")
        visitor = FileAudit(path, source)
        visitor.visit(ast.parse(source, filename=str(path)))
        for category, entries in visitor.findings.items():
            findings[category].extend(entries)
        methods.extend(visitor.methods)

    repeated: dict[tuple[str, str], list[MethodRecord]] = defaultdict(list)
    for method in methods:
        if method.node.name.startswith("__") or _is_trivial_method(method.node):
            continue
        repeated[(method.node.name, _normalized_method(method.node))].append(method)
    duplicate_groups = [
        {
            "method": method_name,
            "body": ast.unparse(_without_docstring(records[0].node.body)),
            "occurrences": [asdict(record.location) for record in records],
        }
        for (method_name, _normalized), records in repeated.items()
        if len({(record.path, record.class_name) for record in records}) > 1
    ]
    duplicate_groups.sort(
        key=lambda group: (
            str(group["method"]),
            str(group["occurrences"][0]["path"]),
            int(group["occurrences"][0]["line"]),
        )
    )

    serialized_findings = {
        category: [
            asdict(entry)
            for entry in sorted(
                entries,
                key=lambda item: (item.path, item.line, item.column, item.name),
            )
        ]
        for category, entries in sorted(findings.items())
    }
    for category in (
        "deleted_api",
        "forwarding_override",
        "mirror_declaration",
        "semantic_reflection",
        "semantic_string_compare",
        "stale_runtime_input_request_api",
    ):
        serialized_findings.setdefault(category, [])
    return {
        "files_scanned": len(scanned_paths),
        "paths": [str(path.relative_to(PROJECT_ROOT)) for path in scanned_paths],
        "findings": serialized_findings,
        "repeated_identical_methods": duplicate_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument(
        "--fail-on-deleted",
        action="store_true",
        help="exit unsuccessfully when a deleted API identifier remains",
    )
    args = parser.parse_args()
    roots = tuple(path.resolve() for path in args.roots)
    report = audit(_source_paths(roots))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_deleted and (
        report["findings"]["deleted_api"]
        or report["findings"]["stale_runtime_input_request_api"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
