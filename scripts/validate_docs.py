#!/usr/bin/env python3
"""Validate active OpenHCS documentation without importing application modules.

The validator uses Python's AST for code examples and filesystem ownership maps
for first-party imports. Sphinx remains the authority for reStructuredText
structure and cross-reference resolution.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import textwrap


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_ROOT = REPOSITORY_ROOT / "docs" / "source"

FIRST_PARTY_MODULE_ROOTS = {
    "openhcs": REPOSITORY_ROOT / "openhcs",
    "objectstate": REPOSITORY_ROOT / "external" / "ObjectState" / "src" / "objectstate",
    "arraybridge": REPOSITORY_ROOT / "external" / "arraybridge" / "src" / "arraybridge",
    "metaclass_registry": (
        REPOSITORY_ROOT / "external" / "metaclass-registry" / "src" / "metaclass_registry"
    ),
    "polystore": REPOSITORY_ROOT / "external" / "PolyStore" / "src" / "polystore",
    "pyqt_reactive": (
        REPOSITORY_ROOT / "external" / "pyqt-reactive" / "src" / "pyqt_reactive"
    ),
    "python_introspect": (
        REPOSITORY_ROOT / "external" / "python-introspect" / "src" / "python_introspect"
    ),
    "zmqruntime": REPOSITORY_ROOT / "external" / "zmqruntime" / "src" / "zmqruntime",
    "pycodify": REPOSITORY_ROOT / "external" / "pycodify" / "src" / "pycodify",
}

PYTHON_DIRECTIVE = re.compile(r"^(?P<indent>\s*)\.\.\s+(?:code-block|code)::\s+python3?\s*$")
MARKDOWN_FENCE = re.compile(r"^\s*```(?:python|py)\s*$", re.IGNORECASE)
LITERAL_INCLUDE = re.compile(r"^\s*\.\.\s+literalinclude::\s+(?P<target>\S+)\s*$")
REPOSITORY_SOURCE_PATH = re.compile(
    r"(?P<target>(?:benchmark|docs|external|openhcs|scripts)/[A-Za-z0-9_.\-/]+"
    r"\.(?:json|md|py|rst|toml|yaml|yml))"
)

FORBIDDEN_IMPORTED_SYMBOLS = {
    "BackendRegistry": "PolyStore uses an explicit backend mapping",
    "analyze_function": "use SignatureAnalyzer().analyze",
}
FORBIDDEN_CALL_NAMES = {
    "Pipeline": "pipelines are list[FunctionStep] plus PipelineConfig",
    "run_pipeline": "compile explicitly and execute a CompiledExecutionBundle",
    "send_request": "ZMQRuntime has no generic send_request API",
}
FORBIDDEN_FUNCTION_STEP_KEYWORDS = {
    "variable_components",
    "group_by",
    "input_source",
}
ARRAYBRIDGE_DECORATORS = {
    "cupy",
    "jax",
    "numpy",
    "pyclesperanto",
    "tensorflow",
    "torch",
}
FORBIDDEN_ARRAYBRIDGE_DECORATOR_KEYWORDS = {
    "clear_cuda_cache",
    "gpu_id",
}


@dataclass(frozen=True)
class CodeBlock:
    path: Path
    line: int
    source: str


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str

    def render(self) -> str:
        relative = self.path.relative_to(REPOSITORY_ROOT)
        return f"{relative}:{self.line}: {self.message}"


def documentation_files(doc_root: Path) -> tuple[Path, ...]:
    if doc_root.is_file():
        return (doc_root,) if doc_root.suffix.lower() in {".rst", ".md", ".json"} else ()
    return tuple(
        sorted(
            path
            for path in doc_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".rst", ".md", ".json"}
            and not {"archive", "plans"}.intersection(
                path.relative_to(doc_root).parts
            )
        )
    )


def rst_python_blocks(path: Path, text: str) -> list[CodeBlock]:
    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    index = 0
    while index < len(lines):
        match = PYTHON_DIRECTIVE.match(lines[index])
        if match is None:
            index += 1
            continue

        directive_indent = len(match.group("indent"))
        cursor = index + 1
        while cursor < len(lines) and (
            not lines[cursor].strip()
            or lines[cursor].lstrip().startswith(":")
        ):
            cursor += 1

        code_start = cursor
        code_lines: list[str] = []
        while cursor < len(lines):
            line = lines[cursor]
            if not line.strip():
                code_lines.append(line)
                cursor += 1
                continue
            indentation = len(line) - len(line.lstrip())
            if indentation <= directive_indent:
                break
            code_lines.append(line)
            cursor += 1

        source = textwrap.dedent("\n".join(code_lines)).strip("\n")
        if source:
            blocks.append(CodeBlock(path=path, line=code_start + 1, source=source))
        index = max(cursor, index + 1)
    return blocks


def markdown_python_blocks(path: Path, text: str) -> list[CodeBlock]:
    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    index = 0
    while index < len(lines):
        if MARKDOWN_FENCE.match(lines[index]) is None:
            index += 1
            continue
        start = index + 1
        index += 1
        code_lines: list[str] = []
        while index < len(lines) and not lines[index].lstrip().startswith("```"):
            code_lines.append(lines[index])
            index += 1
        source = textwrap.dedent("\n".join(code_lines)).strip("\n")
        if source:
            blocks.append(CodeBlock(path=path, line=start + 1, source=source))
        index += 1
    return blocks


def python_blocks(path: Path, text: str) -> list[CodeBlock]:
    if path.suffix.lower() == ".rst":
        return rst_python_blocks(path, text)
    if path.suffix.lower() == ".md":
        return markdown_python_blocks(path, text)
    return []


def module_exists(module_name: str) -> bool:
    parts = module_name.split(".")
    root = FIRST_PARTY_MODULE_ROOTS.get(parts[0])
    if root is None:
        return True
    if len(parts) == 1:
        return root.is_dir()
    candidate = root.joinpath(*parts[1:])
    return candidate.with_suffix(".py").is_file() or (
        candidate.is_dir() and (candidate / "__init__.py").is_file()
    )


def imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def is_step_plan_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "step_plan"
    if isinstance(node, ast.Attribute):
        return node.attr == "step_plan"
    return False


def validate_ast(block: CodeBlock, tree: ast.AST) -> list[Finding]:
    findings: list[Finding] = []
    for module_name in sorted(imported_modules(tree)):
        if not module_exists(module_name):
            findings.append(
                Finding(
                    block.path,
                    block.line,
                    f"first-party import module does not exist: {module_name}",
                )
            )

    for node in ast.walk(tree):
        line = block.line + getattr(node, "lineno", 1) - 1
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                reason = FORBIDDEN_IMPORTED_SYMBOLS.get(alias.name)
                if reason:
                    findings.append(
                        Finding(block.path, line, f"obsolete import {alias.name}: {reason}")
                    )
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            reason = FORBIDDEN_CALL_NAMES.get(name or "")
            if reason:
                findings.append(Finding(block.path, line, f"obsolete call {name}: {reason}"))
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"undo", "redo"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ObjectStateRegistry"
            ):
                replacement = (
                    "time_travel_back"
                    if node.func.attr == "undo"
                    else "time_travel_forward"
                )
                findings.append(
                    Finding(
                        block.path,
                        line,
                        f"obsolete ObjectStateRegistry.{node.func.attr}: use {replacement}",
                    )
                )
            if name == "FunctionStep":
                invalid = sorted(
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg in FORBIDDEN_FUNCTION_STEP_KEYWORDS
                )
                if invalid:
                    findings.append(
                        Finding(
                            block.path,
                            line,
                            "FunctionStep processing fields belong in processing_config: "
                            + ", ".join(invalid),
                        )
                    )
            if name in ARRAYBRIDGE_DECORATORS:
                invalid = sorted(
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg in FORBIDDEN_ARRAYBRIDGE_DECORATOR_KEYWORDS
                )
                if invalid:
                    findings.append(
                        Finding(
                            block.path,
                            line,
                            f"ArrayBridge @{name} decorator does not accept: "
                            + ", ".join(invalid),
                        )
                    )
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and is_step_plan_expression(node.func.value)
            ):
                findings.append(
                    Finding(block.path, line, "string-keyed step_plan.get is obsolete")
                )
        if isinstance(node, ast.Subscript) and is_step_plan_expression(node.value):
            findings.append(Finding(block.path, line, "string-keyed step_plan access is obsolete"))
    return findings


def validate_code_block(block: CodeBlock) -> list[Finding]:
    try:
        tree = ast.parse(block.source)
    except SyntaxError as error:
        line = block.line + (error.lineno or 1) - 1
        return [Finding(block.path, line, f"Python example does not parse: {error.msg}")]
    return validate_ast(block, tree)


def validate_literal_includes(path: Path, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = LITERAL_INCLUDE.match(line)
        if match is None:
            continue
        target = (path.parent / match.group("target")).resolve()
        if not target.is_file():
            findings.append(
                Finding(path, line_number, f"literalinclude target does not exist: {target}")
            )
    return findings


def validate_repository_source_paths(path: Path, text: str) -> list[Finding]:
    """Check concrete repository source paths written in active prose."""
    findings: list[Finding] = []
    relative_parts = path.relative_to(REPOSITORY_ROOT).parts
    owner_root = (
        REPOSITORY_ROOT.joinpath(*relative_parts[:2])
        if len(relative_parts) >= 2 and relative_parts[0] == "external"
        else None
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in REPOSITORY_SOURCE_PATH.finditer(line):
            relative_target = Path(match.group("target"))
            candidates = [REPOSITORY_ROOT / relative_target]
            if owner_root is not None:
                candidates.append(owner_root / relative_target)
            if not any(candidate.is_file() for candidate in candidates):
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"repository source path does not exist: {match.group('target')}",
                    )
                )
    return findings


def validate(doc_root: Path) -> tuple[list[Finding], int, int]:
    findings: list[Finding] = []
    files = documentation_files(doc_root)
    block_count = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                findings.append(
                    Finding(path, error.lineno, f"JSON does not parse: {error.msg}")
                )
        findings.extend(validate_literal_includes(path, text))
        findings.extend(validate_repository_source_paths(path, text))
        blocks = python_blocks(path, text)
        block_count += len(blocks)
        for block in blocks:
            findings.extend(validate_code_block(block))
    return findings, len(files), block_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "doc_root",
        nargs="?",
        type=Path,
        default=DEFAULT_DOC_ROOT,
        help="active documentation file or root (default: docs/source)",
    )
    args = parser.parse_args(argv)
    doc_root = args.doc_root.resolve()
    if not doc_root.exists() or not (doc_root.is_dir() or doc_root.is_file()):
        parser.error(f"documentation target does not exist: {doc_root}")

    findings, file_count, block_count = validate(doc_root)
    for finding in findings:
        print(finding.render())
    if findings:
        print(
            f"documentation validation failed: {len(findings)} finding(s) "
            f"across {file_count} files and {block_count} Python blocks",
            file=sys.stderr,
        )
        return 1
    print(
        f"documentation validation passed: {file_count} files, "
        f"{block_count} Python blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
