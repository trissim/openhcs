#!/usr/bin/env python3
"""Validate active OpenHCS documentation without importing application modules.

The validator uses Python's AST for code examples and filesystem ownership maps
for first-party imports. Sphinx remains the authority for reStructuredText
structure and cross-reference resolution.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_ROOT = REPOSITORY_ROOT / "docs" / "source"
DEFAULT_AUDIT_ROOT = REPOSITORY_ROOT / "docs" / "audits"

AUDIT_DIATAXIS_TYPES = {"tutorial", "how-to", "reference", "explanation"}
AUDIT_DISPOSITIONS = {"keep", "revise", "split", "merge", "remove", "redirect"}
AUDIT_REQUIRED_FIELDS = {
    "path",
    "source_sha256",
    "audience",
    "user_need",
    "diataxis",
    "authority",
    "findings",
    "disposition",
    "validation",
}

FIRST_PARTY_MODULE_ROOTS = {
    "openhcs": REPOSITORY_ROOT / "openhcs",
    "objectstate": REPOSITORY_ROOT / "external" / "ObjectState" / "src" / "objectstate",
    "arraybridge": REPOSITORY_ROOT / "external" / "arraybridge" / "src" / "arraybridge",
    "metaclass_registry": (
        REPOSITORY_ROOT
        / "external"
        / "metaclass-registry"
        / "src"
        / "metaclass_registry"
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

PYTHON_DIRECTIVE = re.compile(
    r"^(?P<indent>\s*)\.\.\s+(?:code-block|code)::\s+python3?\s*$"
)
MARKDOWN_FENCE = re.compile(r"^\s*```(?:python|py)\s*$", re.IGNORECASE)
LITERAL_INCLUDE = re.compile(r"^\s*\.\.\s+literalinclude::\s+(?P<target>\S+)\s*$")
REPOSITORY_SOURCE_PATH = re.compile(
    r"(?P<target>(?:benchmark|docs|external|openhcs|scripts)/[A-Za-z0-9_.\-/]+"
    r"\.(?:json|md|py|rst|toml|yaml|yml))"
)
EXTERNAL_URL = re.compile(r"https?://\S+")

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
        return (
            (doc_root,) if doc_root.suffix.lower() in {".rst", ".md", ".json"} else ()
        )
    return tuple(
        sorted(
            path
            for path in doc_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".rst", ".md", ".json"}
            and not {"archive", "plans"}.intersection(path.relative_to(doc_root).parts)
        )
    )


def active_rst_files(doc_root: Path) -> tuple[Path, ...]:
    """Return active RST sources using the validator's archive exclusions."""
    return tuple(
        path for path in documentation_files(doc_root) if path.suffix == ".rst"
    )


def declared_project_readme(repository_root: Path) -> tuple[Path | None, list[Finding]]:
    """Resolve the published project description from its PEP 621 declaration."""
    pyproject_path = repository_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return None, [Finding(pyproject_path, 1, "project metadata does not exist")]
    try:
        metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        return None, [
            Finding(pyproject_path, 1, f"project metadata does not parse: {error}")
        ]

    project_metadata = metadata.get("project")
    if not isinstance(project_metadata, dict):
        return None, [
            Finding(pyproject_path, 1, "project metadata must declare a project table")
        ]
    readme = project_metadata.get("readme")
    if not isinstance(readme, str) or not readme.strip():
        return None, [
            Finding(
                pyproject_path,
                1,
                "project.readme must declare a non-empty repository-relative path",
            )
        ]
    readme_path = Path(readme)
    if readme_path.is_absolute() or ".." in readme_path.parts:
        return None, [
            Finding(
                pyproject_path,
                1,
                f"project.readme must remain inside the repository: {readme}",
            )
        ]
    return repository_root / readme_path, []


def _nonempty_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_documentation_audit(
    doc_root: Path,
    audit_root: Path,
    repository_root: Path | None = None,
) -> tuple[list[Finding], int]:
    """Validate one editorial record for every active documentation surface."""
    repository_root = repository_root or doc_root.parent.parent
    findings: list[Finding] = []
    audit_files = tuple(sorted(audit_root.glob("*.json")))
    if not audit_files:
        return [Finding(audit_root, 1, "documentation audit has no JSON records")], 0

    entries: list[tuple[Path, object]] = []
    for audit_file in audit_files:
        try:
            payload = json.loads(audit_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            findings.append(
                Finding(
                    audit_file, error.lineno, f"audit JSON does not parse: {error.msg}"
                )
            )
            continue
        if not isinstance(payload, list):
            findings.append(
                Finding(audit_file, 1, "audit payload must be a JSON array")
            )
            continue
        entries.extend((audit_file, entry) for entry in payload)

    project_readme, readme_findings = declared_project_readme(repository_root)
    findings.extend(readme_findings)
    audited_sources = list(active_rst_files(doc_root))
    if project_readme is not None:
        audited_sources.append(project_readme)
    expected_paths = {
        path.relative_to(repository_root).as_posix() for path in audited_sources
    }
    seen: dict[str, Path] = {}
    for audit_file, entry in entries:
        if not isinstance(entry, dict):
            findings.append(Finding(audit_file, 1, "audit entry must be a JSON object"))
            continue
        missing_fields = sorted(AUDIT_REQUIRED_FIELDS - set(entry))
        extra_fields = sorted(set(entry) - AUDIT_REQUIRED_FIELDS)
        if missing_fields:
            findings.append(
                Finding(
                    audit_file,
                    1,
                    "audit entry is missing fields: " + ", ".join(missing_fields),
                )
            )
        if extra_fields:
            findings.append(
                Finding(
                    audit_file,
                    1,
                    "audit entry has unsupported fields: " + ", ".join(extra_fields),
                )
            )

        source_path = entry.get("path")
        if not isinstance(source_path, str) or not source_path.strip():
            findings.append(
                Finding(audit_file, 1, "audit path must be a non-empty string")
            )
            continue
        if source_path in seen:
            findings.append(
                Finding(
                    audit_file,
                    1,
                    f"duplicate audit path {source_path}; first declared in {seen[source_path]}",
                )
            )
        else:
            seen[source_path] = audit_file

        source_digest = entry.get("source_sha256")
        source_file = repository_root / source_path
        if not _valid_sha256(source_digest):
            findings.append(
                Finding(
                    audit_file,
                    1,
                    f"{source_path}: source_sha256 must be a lowercase digest",
                )
            )
        elif source_path in expected_paths:
            if not source_file.is_file():
                findings.append(
                    Finding(
                        audit_file,
                        1,
                        f"{source_path}: audited documentation source does not exist",
                    )
                )
            else:
                actual_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
                if source_digest != actual_digest:
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: source changed after its editorial audit",
                        )
                    )

        if not _nonempty_string_list(entry.get("audience")):
            findings.append(
                Finding(
                    audit_file,
                    1,
                    f"{source_path}: audience must be a non-empty string list",
                )
            )
        user_need = entry.get("user_need")
        if not isinstance(user_need, str) or not user_need.strip():
            findings.append(
                Finding(
                    audit_file,
                    1,
                    f"{source_path}: user_need must be a non-empty string",
                )
            )
        diataxis = entry.get("diataxis")
        if diataxis not in AUDIT_DIATAXIS_TYPES:
            findings.append(
                Finding(
                    audit_file, 1, f"{source_path}: invalid Diataxis type {diataxis!r}"
                )
            )
        authorities = entry.get("authority")
        if not isinstance(authorities, list) or not authorities:
            findings.append(
                Finding(
                    audit_file, 1, f"{source_path}: authority must be a non-empty list"
                )
            )
        else:
            has_non_documentation_authority = False
            for authority in authorities:
                if not isinstance(authority, dict):
                    findings.append(
                        Finding(
                            audit_file, 1, f"{source_path}: authority must be an object"
                        )
                    )
                    continue
                if set(authority) != {"path", "sha256", "role"}:
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: authority requires path, sha256, and role",
                        )
                    )
                    continue
                authority_relative = authority["path"]
                authority_role = authority["role"]
                authority_digest = authority["sha256"]
                if (
                    not isinstance(authority_relative, str)
                    or not authority_relative.strip()
                ):
                    findings.append(
                        Finding(
                            audit_file, 1, f"{source_path}: authority path is empty"
                        )
                    )
                    continue
                authority_relative_path = Path(authority_relative)
                if (
                    authority_relative_path.is_absolute()
                    or ".." in authority_relative_path.parts
                    or authority_relative_path.as_posix() != authority_relative
                ):
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: authority path must be canonical and "
                            f"repository-relative: {authority_relative}",
                        )
                    )
                    continue
                if not isinstance(authority_role, str) or not authority_role.strip():
                    findings.append(
                        Finding(
                            audit_file, 1, f"{source_path}: authority role is empty"
                        )
                    )
                elif isinstance(user_need, str) and user_need in authority_role:
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: authority role repeats the user need "
                            "instead of naming file-specific evidence",
                        )
                    )
                authority_path = (repository_root / authority_relative_path).resolve()
                try:
                    authority_path.relative_to(repository_root.resolve())
                except ValueError:
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: authority escapes the repository: "
                            f"{authority_relative}",
                        )
                    )
                    continue
                if not authority_path.is_file():
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: authority file does not exist: "
                            f"{authority_relative}",
                        )
                    )
                    continue
                if not _valid_sha256(authority_digest):
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: authority sha256 is invalid: "
                            f"{authority_relative}",
                        )
                    )
                    continue
                actual_authority_digest = hashlib.sha256(
                    authority_path.read_bytes()
                ).hexdigest()
                if authority_digest != actual_authority_digest:
                    findings.append(
                        Finding(
                            audit_file,
                            1,
                            f"{source_path}: authority changed after review: "
                            f"{authority_relative}",
                        )
                    )
                if not authority_relative.startswith("docs/source/"):
                    has_non_documentation_authority = True
            if not has_non_documentation_authority:
                findings.append(
                    Finding(
                        audit_file,
                        1,
                        f"{source_path}: another documentation page cannot be its only authority",
                    )
                )
        entry_findings = entry.get("findings")
        if not isinstance(entry_findings, list) or not all(
            isinstance(item, str) and bool(item.strip()) for item in entry_findings
        ):
            findings.append(
                Finding(audit_file, 1, f"{source_path}: findings must be a string list")
            )
        disposition = entry.get("disposition")
        if disposition not in AUDIT_DISPOSITIONS:
            findings.append(
                Finding(
                    audit_file, 1, f"{source_path}: invalid disposition {disposition!r}"
                )
            )
        if disposition != "keep" and entry_findings == []:
            findings.append(
                Finding(
                    audit_file,
                    1,
                    f"{source_path}: a changed disposition requires a recorded finding",
                )
            )
        if not _nonempty_string_list(entry.get("validation")):
            findings.append(
                Finding(
                    audit_file,
                    1,
                    f"{source_path}: validation must be a non-empty string list",
                )
            )

    audited_paths = set(seen)
    for missing_path in sorted(expected_paths - audited_paths):
        findings.append(
            Finding(
                audit_root,
                1,
                f"active documentation source is not audited: {missing_path}",
            )
        )
    for extra_path in sorted(audited_paths - expected_paths):
        findings.append(
            Finding(seen[extra_path], 1, f"audit path is not active: {extra_path}")
        )
    return findings, len(entries)


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
            not lines[cursor].strip() or lines[cursor].lstrip().startswith(":")
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
                        Finding(
                            block.path, line, f"obsolete import {alias.name}: {reason}"
                        )
                    )
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            reason = FORBIDDEN_CALL_NAMES.get(name or "")
            if reason:
                findings.append(
                    Finding(block.path, line, f"obsolete call {name}: {reason}")
                )
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
            findings.append(
                Finding(block.path, line, "string-keyed step_plan access is obsolete")
            )
    return findings


def validate_code_block(block: CodeBlock) -> list[Finding]:
    try:
        tree = ast.parse(block.source)
    except SyntaxError as error:
        line = block.line + (error.lineno or 1) - 1
        return [
            Finding(block.path, line, f"Python example does not parse: {error.msg}")
        ]
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
                Finding(
                    path, line_number, f"literalinclude target does not exist: {target}"
                )
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
        external_url_spans = tuple(
            (match.start(), match.end()) for match in EXTERNAL_URL.finditer(line)
        )
        for match in REPOSITORY_SOURCE_PATH.finditer(line):
            if any(
                url_start <= match.start() < url_end
                for url_start, url_end in external_url_spans
            ):
                continue
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


def validate(
    doc_root: Path,
    additional_sources: tuple[Path, ...] = (),
) -> tuple[list[Finding], int, int]:
    findings: list[Finding] = []
    files = (*documentation_files(doc_root), *additional_sources)
    block_count = 0
    for path in files:
        if not path.is_file():
            findings.append(Finding(path, 1, "documentation source does not exist"))
            continue
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

    project_readme: Path | None = None
    if doc_root == DEFAULT_DOC_ROOT.resolve():
        project_readme, _ = declared_project_readme(REPOSITORY_ROOT)
    additional_sources = () if project_readme is None else (project_readme,)
    findings, file_count, block_count = validate(
        doc_root,
        additional_sources=additional_sources,
    )
    audit_entry_count: int | None = None
    if doc_root == DEFAULT_DOC_ROOT.resolve():
        audit_findings, audit_entry_count = validate_documentation_audit(
            doc_root,
            DEFAULT_AUDIT_ROOT,
        )
        findings.extend(audit_findings)
    for finding in findings:
        print(finding.render())
    if findings:
        print(
            f"documentation validation failed: {len(findings)} finding(s) "
            f"across {file_count} files and {block_count} Python blocks",
            file=sys.stderr,
        )
        return 1
    audit_summary = (
        f", {audit_entry_count} audited documentation sources"
        if audit_entry_count is not None
        else ""
    )
    print(
        f"documentation validation passed: {file_count} files, "
        f"{block_count} Python blocks{audit_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
