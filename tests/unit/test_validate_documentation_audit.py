"""Tests for complete, non-mirrored documentation audit coverage."""

import hashlib
import json
from pathlib import Path

from scripts.validate_docs import (
    REPOSITORY_ROOT,
    validate,
    validate_documentation_audit,
    validate_repository_source_paths,
)


def _entry(
    path: str,
    source: bytes = b"Index\n=====\n",
    authority_path: str = "src/task.py",
    authority_source: bytes = b"TASK = True\n",
) -> dict[str, object]:
    return {
        "path": path,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "audience": ["operator"],
        "user_need": "Complete one documented task.",
        "diataxis": "how-to",
        "authority": [
            {
                "path": authority_path,
                "sha256": hashlib.sha256(authority_source).hexdigest(),
                "role": "Owns the documented task behaviour.",
            }
        ],
        "findings": [],
        "disposition": "keep",
        "validation": ["Checked against src/task.py."],
    }


def _write_audit(
    audit_root: Path,
    entries: list[dict[str, object]],
    *,
    include_project_readme: bool = True,
) -> None:
    audit_root.mkdir(parents=True)
    complete_entries = list(entries)
    if include_project_readme:
        complete_entries.append(_entry("README.md", source=b"# Project\n"))
    (audit_root / "section.json").write_text(
        json.dumps(complete_entries),
        encoding="utf-8",
    )


def _write_authority(project_root: Path, source: bytes = b"TASK = True\n") -> None:
    authority = project_root / "src" / "task.py"
    authority.parent.mkdir(parents=True)
    authority.write_bytes(source)
    (project_root / "README.md").write_bytes(b"# Project\n")
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "project"\nversion = "1.0.0"\nreadme = "README.md"\n',
        encoding="utf-8",
    )


def test_audit_requires_each_active_rst_and_declared_project_readme(
    tmp_path: Path,
) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    first_page = doc_root / "index.rst"
    second_page = doc_root / "guide.rst"
    first_page.write_text("Index\n=====\n", encoding="utf-8")
    second_page.write_text("Guide\n=====\n", encoding="utf-8")
    _write_authority(tmp_path)
    _write_audit(
        audit_root,
        [
            _entry(first_page.relative_to(tmp_path).as_posix()),
            _entry(
                second_page.relative_to(tmp_path).as_posix(),
                b"Guide\n=====\n",
            ),
        ],
    )

    findings, count = validate_documentation_audit(doc_root, audit_root)

    assert findings == []
    assert count == 3


def test_audit_rejects_missing_duplicate_and_inactive_paths(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    active_page = doc_root / "index.rst"
    active_page.write_text("Index\n=====\n", encoding="utf-8")
    _write_authority(tmp_path)
    inactive_path = "docs/source/archive/old.rst"
    duplicate = _entry(inactive_path)
    _write_audit(audit_root, [duplicate, duplicate])

    findings, _ = validate_documentation_audit(doc_root, audit_root)
    messages = [finding.message for finding in findings]

    assert any("duplicate audit path" in message for message in messages)
    assert any(
        "active documentation source is not audited" in message for message in messages
    )
    assert any("audit path is not active" in message for message in messages)


def test_audit_rejects_invalid_editorial_evidence(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    page = doc_root / "index.rst"
    page.write_text("Index\n=====\n", encoding="utf-8")
    _write_authority(tmp_path)
    entry = _entry(page.relative_to(tmp_path).as_posix())
    entry["audience"] = []
    entry["source_sha256"] = "not-a-digest"
    entry["user_need"] = ""
    entry["diataxis"] = "overview"
    entry["authority"] = []
    entry["findings"] = [""]
    entry["disposition"] = "ignore"
    entry["validation"] = []
    _write_audit(audit_root, [entry])

    findings, _ = validate_documentation_audit(doc_root, audit_root)
    messages = [finding.message for finding in findings]

    assert any("audience" in message for message in messages)
    assert any("source_sha256" in message for message in messages)
    assert any("user_need" in message for message in messages)
    assert any("Diataxis" in message for message in messages)
    assert any("authority" in message for message in messages)
    assert any("findings" in message for message in messages)
    assert any("disposition" in message for message in messages)
    assert any("validation" in message for message in messages)


def test_audit_rejects_source_changed_after_review(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    page = doc_root / "index.rst"
    page.write_text("Changed\n=======\n", encoding="utf-8")
    _write_authority(tmp_path)
    _write_audit(
        audit_root,
        [_entry(page.relative_to(tmp_path).as_posix())],
    )

    findings, _ = validate_documentation_audit(doc_root, audit_root)

    assert any("source changed after" in finding.message for finding in findings)


def test_audit_requires_the_project_readme_declared_by_package_metadata(
    tmp_path: Path,
) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    page = doc_root / "index.rst"
    page.write_text("Index\n=====\n", encoding="utf-8")
    _write_authority(tmp_path)
    _write_audit(
        audit_root,
        [_entry(page.relative_to(tmp_path).as_posix())],
        include_project_readme=False,
    )

    findings, _ = validate_documentation_audit(doc_root, audit_root)

    assert any(
        finding.message == "active documentation source is not audited: README.md"
        for finding in findings
    )


def test_audit_rejects_changed_project_readme(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    page = doc_root / "index.rst"
    page.write_text("Index\n=====\n", encoding="utf-8")
    _write_authority(tmp_path)
    _write_audit(audit_root, [_entry(page.relative_to(tmp_path).as_posix())])
    (tmp_path / "README.md").write_text("# Changed project\n", encoding="utf-8")

    findings, _ = validate_documentation_audit(doc_root, audit_root)

    assert any(
        finding.message == "README.md: source changed after its editorial audit"
        for finding in findings
    )


def test_project_readme_python_blocks_receive_documentation_validation() -> None:
    findings, file_count, block_count = validate(
        REPOSITORY_ROOT / "docs" / "source" / "index.rst",
        additional_sources=(REPOSITORY_ROOT / "README.md",),
    )

    assert file_count == 2
    assert block_count == 1
    assert findings == []


def test_missing_additional_documentation_source_is_a_finding(
    tmp_path: Path,
) -> None:
    missing_readme = tmp_path / "README.md"

    findings, file_count, block_count = validate(
        REPOSITORY_ROOT / "docs" / "source" / "index.rst",
        additional_sources=(missing_readme,),
    )

    assert file_count == 2
    assert block_count == 0
    assert len(findings) == 1
    assert findings[0].path == missing_readme
    assert findings[0].message == "documentation source does not exist"


def test_audit_rejects_authority_changed_after_review(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    page = doc_root / "index.rst"
    page.write_text("Index\n=====\n", encoding="utf-8")
    _write_authority(tmp_path, b"TASK = False\n")
    _write_audit(
        audit_root,
        [_entry(page.relative_to(tmp_path).as_posix())],
    )

    findings, _ = validate_documentation_audit(doc_root, audit_root)

    assert any("authority changed after" in finding.message for finding in findings)


def test_audit_rejects_another_doc_as_the_only_authority(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    page = doc_root / "index.rst"
    page.write_text("Index\n=====\n", encoding="utf-8")
    docs_config = doc_root / "conf.py"
    docs_config.write_bytes(b"project = 'Example'\n")
    _write_audit(
        audit_root,
        [
            _entry(
                page.relative_to(tmp_path).as_posix(),
                authority_path=docs_config.relative_to(tmp_path).as_posix(),
                authority_source=docs_config.read_bytes(),
            )
        ],
    )

    findings, _ = validate_documentation_audit(doc_root, audit_root)

    assert any(
        "cannot be its only authority" in finding.message for finding in findings
    )


def test_audit_rejects_authority_role_that_only_repeats_the_user_need(
    tmp_path: Path,
) -> None:
    doc_root = tmp_path / "docs" / "source"
    audit_root = tmp_path / "docs" / "audits"
    doc_root.mkdir(parents=True)
    page = doc_root / "index.rst"
    page.write_text("Index\n=====\n", encoding="utf-8")
    _write_authority(tmp_path)
    entry = _entry(page.relative_to(tmp_path).as_posix())
    entry["authority"][0]["role"] = entry["user_need"]
    _write_audit(audit_root, [entry])

    findings, _ = validate_documentation_audit(doc_root, audit_root)

    assert any("repeats the user need" in finding.message for finding in findings)


def test_repository_source_path_validation_leaves_external_urls_to_linkcheck() -> None:
    source_page = REPOSITORY_ROOT / "docs" / "source" / "index.rst"
    target = "docs/source/architecture/not_a_local_page.rst"

    assert (
        validate_repository_source_paths(
            source_page,
            f"https://github.com/example/project/blob/v1/{target}",
        )
        == []
    )
    assert len(validate_repository_source_paths(source_page, target)) == 1
